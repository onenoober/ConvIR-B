#!/usr/bin/env python3
"""Build one typed route asset manifest from an explicit identity request."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from route_runtime_contract import (
    ASSET_ACCESS_ROLES,
    ContractError,
    SHA40,
    SHA256,
    require_asset_path,
    require_bool,
    require_token,
    validate_asset_manifest,
    validate_runtime_spec,
)


GIT = "/usr/bin/git"
LOCAL_IDENTITY_ROLES = {"unrestricted", "engineering_debug"}
REQUEST_FIELDS = {"schema_version", "assets"}
ASSET_FIELDS = {"id", "kind", "path", "access_role", "contract_access", "identity"}


class AssetBuildError(RuntimeError):
    pass


def load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssetBuildError(f"cannot read {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise AssetBuildError(f"{name} must be a JSON object")
    return value


def sha256_file(path: Path, maximum_bytes: int) -> str:
    if not path.is_file():
        raise AssetBuildError(f"identity source is not a file: {path}")
    size = path.stat().st_size
    if not 0 <= size <= maximum_bytes:
        raise AssetBuildError(f"identity source exceeds hash budget: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_identity(path: Path, require_clean: bool) -> str:
    if not path.is_dir():
        raise AssetBuildError(f"identity source is not a checkout: {path}")
    completed = subprocess.run(
        [GIT, "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
        text=True, timeout=30, check=False,
    )
    commit = completed.stdout.strip()
    if completed.returncode or not SHA40.fullmatch(commit):
        raise AssetBuildError(f"cannot resolve checkout identity: {path}")
    if require_clean:
        status = subprocess.run(
            [GIT, "-C", str(path), "status", "--porcelain"], capture_output=True,
            text=True, timeout=30, check=False,
        )
        if status.returncode or status.stdout.strip():
            raise AssetBuildError(f"identity checkout is not clean: {path}")
    return commit


def build_asset(item: dict[str, Any], maximum_hash_bytes: int) -> dict[str, Any]:
    if not isinstance(item, dict) or set(item) != ASSET_FIELDS:
        raise AssetBuildError("asset request has an invalid field contract")
    identifier = require_token(item["id"], "asset.id")
    kind = item["kind"]
    if kind not in {"file", "directory", "git_checkout"}:
        raise AssetBuildError(f"unsupported asset kind: {kind}")
    path = require_asset_path(item["path"], f"asset.{identifier}.path")
    role = item["access_role"]
    if role not in ASSET_ACCESS_ROLES:
        raise AssetBuildError(f"invalid asset access role: {role}")
    contract_access = require_bool(item["contract_access"], "asset.contract_access")
    identity = item["identity"]
    if not isinstance(identity, dict):
        raise AssetBuildError("asset identity must be an object")
    result = {
        "id": identifier, "kind": kind, "path": path,
        "access_role": role, "contract_access": contract_access,
    }
    if kind == "directory":
        if identity:
            raise AssetBuildError("directory identity must be empty")
        return result
    if kind == "file":
        if set(identity) == {"sha256"}:
            digest = identity["sha256"]
            if not isinstance(digest, str) or not SHA256.fullmatch(digest):
                raise AssetBuildError("predeclared file sha256 is invalid")
        elif set(identity) == {"local_file"}:
            if role not in LOCAL_IDENTITY_ROLES:
                raise AssetBuildError(
                    "local hashing is limited to unrestricted/engineering assets"
                )
            if not isinstance(identity["local_file"], str):
                raise AssetBuildError("local_file must be a string")
            source = Path(identity["local_file"])
            if not source.is_absolute():
                raise AssetBuildError("local_file must be absolute")
            digest = sha256_file(source.resolve(), maximum_hash_bytes)
        else:
            raise AssetBuildError("file identity needs exactly sha256 or local_file")
        result["sha256"] = digest
        return result
    require_clean = identity.get("require_clean")
    require_bool(require_clean, "asset.require_clean")
    if set(identity) == {"commit", "require_clean"}:
        commit = identity["commit"]
        if not isinstance(commit, str) or not SHA40.fullmatch(commit):
            raise AssetBuildError("predeclared checkout commit is invalid")
    elif set(identity) == {"local_checkout", "require_clean"}:
        if role not in LOCAL_IDENTITY_ROLES:
            raise AssetBuildError(
                "local checkout inspection is limited to unrestricted/engineering assets"
            )
        if not isinstance(identity["local_checkout"], str):
            raise AssetBuildError("local_checkout must be a string")
        source = Path(identity["local_checkout"])
        if not source.is_absolute():
            raise AssetBuildError("local_checkout must be absolute")
        commit = git_identity(source.resolve(), require_clean)
    else:
        raise AssetBuildError(
            "git identity needs commit or local_checkout plus require_clean"
        )
    result.update({"commit": commit, "require_clean": require_clean})
    return result


def build_manifest(spec: dict[str, Any], request: dict[str, Any],
                   maximum_hash_bytes: int) -> dict[str, Any]:
    if not isinstance(maximum_hash_bytes, int) or isinstance(maximum_hash_bytes, bool) \
            or maximum_hash_bytes < 1:
        raise AssetBuildError("maximum_hash_bytes must be a positive integer")
    if not isinstance(request, dict) or set(request) != REQUEST_FIELDS \
            or request.get("schema_version") != 1:
        raise AssetBuildError("asset build request must use schema 1")
    assets = request["assets"]
    if not isinstance(assets, list) or not 1 <= len(assets) <= 128:
        raise AssetBuildError("asset request must contain 1-128 assets")
    built = [build_asset(item, maximum_hash_bytes) for item in assets]
    if len({item["id"] for item in built}) != len(built):
        raise AssetBuildError("asset request ids must be unique")
    manifest = {
        "schema_version": 1,
        "route_id": require_token(spec["route_id"], "route_id"),
        "operation_id": require_token(spec["operation_id"], "operation_id"),
        "assets": built,
    }
    try:
        return validate_asset_manifest(manifest, spec)
    except ContractError as exc:
        raise AssetBuildError(str(exc)) from exc


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise AssetBuildError(f"refusing to replace existing asset manifest: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(repo: Path, operation_id: str, request_path: Path, *,
            maximum_hash_bytes: int, apply: bool) -> dict[str, Any]:
    try:
        request_path.relative_to(repo)
    except ValueError:
        pass
    else:
        raise AssetBuildError("asset request must stay outside the route worktree")
    manifest = load_json(repo / "experience_docx/route_operations.json", "route manifest")
    spec_path = repo / f"experience_docx/route_runtime_specs/{require_token(operation_id, 'operation_id')}.json"
    raw_spec = load_json(spec_path, "runtime spec")
    try:
        spec = validate_runtime_spec(raw_spec, manifest, operation_id)
    except ContractError as exc:
        raise AssetBuildError(str(exc)) from exc
    asset_relpath = spec["asset_manifest_relpath"]
    if asset_relpath is None:
        raise AssetBuildError("runtime spec does not declare an asset manifest")
    request = load_json(request_path, "asset request")
    asset_manifest = build_manifest(spec, request, maximum_hash_bytes)
    output = repo / asset_relpath
    if apply:
        atomic_write(output, asset_manifest)
    digest = hashlib.sha256(
        json.dumps(asset_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "status": "ASSET_MANIFEST_APPLIED" if apply else "ASSET_MANIFEST_READY",
        "route_id": spec["route_id"], "operation_id": operation_id,
        "asset_manifest_relpath": asset_relpath,
        "asset_count": len(asset_manifest["assets"]),
        "asset_ids": [item["id"] for item in asset_manifest["assets"]],
        "canonical_sha256": digest,
        "local_content_reads": sum(
            int("local_file" in item["identity"] or "local_checkout" in item["identity"])
            for item in request["assets"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--operation", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--max-hash-bytes", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_hash_bytes <= 16 * 1024 * 1024 * 1024:
        print("ASSET_MANIFEST_ERROR max-hash-bytes is outside limits")
        raise SystemExit(1)
    try:
        report = prepare(
            args.repo.resolve(), args.operation, args.request.resolve(),
            maximum_hash_bytes=args.max_hash_bytes, apply=args.apply,
        )
    except (AssetBuildError, ContractError) as exc:
        print(f"ASSET_MANIFEST_ERROR {exc}")
        raise SystemExit(1)
    print(json.dumps(report, sort_keys=True))
    print(f"{report['status']} operation_id={report['operation_id']}")


if __name__ == "__main__":
    main()
