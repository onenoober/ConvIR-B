#!/usr/bin/env python3
"""Validate and query exact-identity engineering capability qualifications."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Iterable


class CapabilityRegistryError(RuntimeError):
    pass


REGISTRY_RELPATH = "experience_docx/capability_registry.jsonl"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
IDENTITY_FIELDS = {
    "source_commit", "code_path_sha256", "checkpoint_sha256",
    "runtime_environment_sha256", "device_class", "input_contract_sha256",
}


def canonical_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def identity_digest(identity: dict) -> str:
    validate_identity(identity)
    return hashlib.sha256(canonical_bytes(identity)).hexdigest()


def validate_identity(value) -> dict:
    if not isinstance(value, dict) or set(value) != IDENTITY_FIELDS:
        raise CapabilityRegistryError("capability identity has an invalid field contract")
    if not SHA40.fullmatch(value["source_commit"]):
        raise CapabilityRegistryError("source_commit is invalid")
    for key in IDENTITY_FIELDS - {"source_commit", "device_class"}:
        if not isinstance(value[key], str) or not SHA256.fullmatch(value[key]):
            raise CapabilityRegistryError(f"{key} is invalid")
    if not isinstance(value["device_class"], str) or not SAFE_TOKEN.fullmatch(value["device_class"]):
        raise CapabilityRegistryError("device_class is invalid")
    return dict(value)


def validate_record(value, *, evidence_exists: Callable[[str], bool] | None = None,
                    read_evidence: Callable[[str], bytes] | None = None) -> dict:
    expected = {
        "schema_version", "qualification_id", "identity", "identity_sha256",
        "status", "contract_mode", "evidence_relpath", "evidence_sha256",
        "scientific_authorization", "protected_data_touched",
    }
    if not isinstance(value, dict) or set(value) != expected or value["schema_version"] != 1:
        raise CapabilityRegistryError("capability record has an invalid field contract")
    if not isinstance(value["qualification_id"], str) or not SAFE_TOKEN.fullmatch(value["qualification_id"]):
        raise CapabilityRegistryError("qualification_id is invalid")
    identity = validate_identity(value["identity"])
    digest = identity_digest(identity)
    if value["identity_sha256"] != digest:
        raise CapabilityRegistryError("identity_sha256 mismatch")
    if value["status"] != "PASSED_ENGINEERING":
        raise CapabilityRegistryError("only passed engineering qualifications are reusable")
    if not isinstance(value["contract_mode"], str) or not SAFE_TOKEN.fullmatch(value["contract_mode"]):
        raise CapabilityRegistryError("contract_mode is invalid")
    relpath = value["evidence_relpath"]
    if not isinstance(relpath, str) or Path(relpath).is_absolute() \
            or ".." in Path(relpath).parts or not relpath.startswith("experience_docx/"):
        raise CapabilityRegistryError("evidence_relpath is invalid")
    if evidence_exists is not None and not evidence_exists(relpath):
        raise CapabilityRegistryError("qualification evidence is missing")
    if not isinstance(value["evidence_sha256"], str) or not SHA256.fullmatch(value["evidence_sha256"]):
        raise CapabilityRegistryError("evidence_sha256 is invalid")
    if read_evidence is not None:
        try:
            raw = read_evidence(relpath)
        except (FileNotFoundError, KeyError) as exc:
            raise CapabilityRegistryError("qualification evidence is missing") from exc
        if hashlib.sha256(raw).hexdigest() != value["evidence_sha256"]:
            raise CapabilityRegistryError("qualification evidence SHA-256 mismatch")
    if value["scientific_authorization"] != "NONE":
        raise CapabilityRegistryError("engineering qualification cannot carry scientific authorization")
    if value["protected_data_touched"] is not False:
        raise CapabilityRegistryError("reusable engineering qualification must not touch protected data")
    return {**value, "identity": identity}


def load_records(lines: Iterable[str], *, evidence_exists: Callable[[str], bool] | None = None,
                 read_evidence: Callable[[str], bytes] | None = None) -> list[dict]:
    records = []
    ids = set()
    identities = set()
    for number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CapabilityRegistryError(f"invalid JSONL record at line {number}") from exc
        record = validate_record(
            value, evidence_exists=evidence_exists, read_evidence=read_evidence,
        )
        if record["qualification_id"] in ids:
            raise CapabilityRegistryError("duplicate qualification_id")
        if record["identity_sha256"] in identities:
            raise CapabilityRegistryError("duplicate capability identity")
        ids.add(record["qualification_id"])
        identities.add(record["identity_sha256"])
        records.append(record)
    return records


def lookup(records: Iterable[dict], identity: dict) -> dict:
    digest = identity_digest(identity)
    matches = [record for record in records if record.get("identity_sha256") == digest]
    if len(matches) != 1:
        return {
            "status": "CAPABILITY_REUSE_MISS",
            "identity_sha256": digest,
            "engineering_reuse_authorized": False,
            "scientific_authorization": "NONE",
        }
    record = validate_record(matches[0])
    return {
        "status": "CAPABILITY_REUSE_EXACT_MATCH",
        "identity_sha256": digest,
        "qualification_id": record["qualification_id"],
        "evidence_relpath": record["evidence_relpath"],
        "evidence_sha256": record["evidence_sha256"],
        "engineering_reuse_authorized": True,
        "scientific_authorization": "NONE",
    }


def lookup_lines(
    lines: Iterable[str], identity: dict, *,
    evidence_exists: Callable[[str], bool] | None = None,
    read_evidence: Callable[[str], bytes] | None = None,
) -> dict:
    """Validate only the unique record matching one requested identity.

    Route readiness and lifecycle reuse are point queries. Unrelated historical
    records are maintained by registry CI/archive validation and must not make a
    reuse miss fail. A matching record remains fail-closed, including duplicate
    matches and its exact evidence binding.
    """
    digest = identity_digest(identity)
    matches = []
    for raw in lines:
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("identity_sha256") == digest:
            matches.append(value)
    if not matches:
        return {
            "status": "CAPABILITY_REUSE_MISS",
            "identity_sha256": digest,
            "engineering_reuse_authorized": False,
            "scientific_authorization": "NONE",
        }
    if len(matches) != 1:
        raise CapabilityRegistryError("duplicate matching capability identity")
    record = validate_record(
        matches[0], evidence_exists=evidence_exists, read_evidence=read_evidence,
    )
    return lookup([record], identity)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--registry", default=REGISTRY_RELPATH)
    parser.add_argument("--identity", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    registry_path = repo / args.registry
    identity_path = args.identity.resolve()
    try:
        identity_path.relative_to(repo)
    except ValueError as exc:
        raise SystemExit("identity file must stay inside the repository") from exc
    records = load_records(
        registry_path.read_text(encoding="utf-8").splitlines(),
        evidence_exists=lambda relpath: (repo / relpath).is_file(),
        read_evidence=lambda relpath: (repo / relpath).read_bytes(),
    )
    value = lookup(records, json.loads(identity_path.read_text(encoding="utf-8")))
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
