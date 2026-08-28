#!/usr/bin/env python3
"""Fixed dataset-registry resolution for experiment-assistant contracts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

from experiment_assistant_contract import (
    DATA_ROLES,
    PROTECTED_DATA_ROLES,
    canonical_sha256,
)


REGISTRY_SCHEMA_VERSION = 1
MAX_REGISTRY_BYTES = 1024 * 1024
MAX_DATASETS = 1024
SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DatasetRegistryError(RuntimeError):
    """The fixed registry cannot reliably bind the requested dataset contract."""


def _read_regular(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DatasetRegistryError("dataset registry cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_REGISTRY_BYTES:
            raise DatasetRegistryError("dataset registry must be a bounded regular file")
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(256 * 1024, MAX_REGISTRY_BYTES + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_REGISTRY_BYTES:
                raise DatasetRegistryError("dataset registry exceeds its bounded size")
            chunks.append(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(data) != after.st_size:
            raise DatasetRegistryError("dataset registry exceeds its bounded size")
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise DatasetRegistryError("dataset registry changed while reading")
        return data
    finally:
        os.close(descriptor)


def _dataset_path(raw: Any, identifier: str) -> str:
    if not isinstance(raw, str) or not raw or len(raw) > 4096:
        raise DatasetRegistryError(f"dataset {identifier} path must be bounded text")
    path = Path(raw)
    if not path.is_absolute():
        raise DatasetRegistryError(f"dataset {identifier} path must be absolute")
    try:
        path_stat = path.lstat()
    except OSError as exc:
        raise DatasetRegistryError(f"dataset {identifier} path is unavailable") from exc
    if stat.S_ISLNK(path_stat.st_mode) or not (
        stat.S_ISREG(path_stat.st_mode) or stat.S_ISDIR(path_stat.st_mode)
    ):
        raise DatasetRegistryError(
            f"dataset {identifier} path must be a regular file or directory"
        )
    return str(path.resolve())


class DatasetRegistry:
    def __init__(self, path: Path):
        path = Path(path)
        if not path.is_absolute():
            raise DatasetRegistryError("dataset registry path must be absolute")
        self.path = path.resolve()
        data = _read_regular(path)
        try:
            source = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatasetRegistryError("dataset registry must be valid UTF-8 JSON") from exc
        if not isinstance(source, dict) or set(source) != {"schema_version", "datasets"} \
                or source.get("schema_version") != REGISTRY_SCHEMA_VERSION:
            raise DatasetRegistryError("dataset registry has an unsupported field contract")
        items = source.get("datasets")
        if not isinstance(items, list) or not 1 <= len(items) <= MAX_DATASETS:
            raise DatasetRegistryError(
                f"dataset registry must contain 1-{MAX_DATASETS} entries"
            )
        entries: dict[str, dict[str, Any]] = {}
        for index, raw in enumerate(items):
            if not isinstance(raw, dict) or set(raw) != {
                "id", "role", "path", "identity_sha256", "protected",
            }:
                raise DatasetRegistryError(f"dataset registry entry {index} is invalid")
            identifier = raw.get("id")
            if not isinstance(identifier, str) or not SAFE_TOKEN.fullmatch(identifier):
                raise DatasetRegistryError(f"dataset registry entry {index} has an invalid id")
            if identifier in entries:
                raise DatasetRegistryError(f"dataset registry id is duplicated: {identifier}")
            role = raw.get("role")
            if role not in DATA_ROLES:
                raise DatasetRegistryError(f"dataset {identifier} has an invalid role")
            protected = raw.get("protected")
            if not isinstance(protected, bool) or protected != (role in PROTECTED_DATA_ROLES):
                raise DatasetRegistryError(
                    f"dataset {identifier} protected flag conflicts with role {role}"
                )
            identity = raw.get("identity_sha256")
            if not isinstance(identity, str) or not SHA256.fullmatch(identity):
                raise DatasetRegistryError(f"dataset {identifier} identity must be SHA-256")
            entries[identifier] = {
                "id": identifier,
                "role": role,
                "path": _dataset_path(raw.get("path"), identifier),
                "identity_sha256": identity,
                "protected": protected,
            }
        self.entries = entries
        self.registry_sha256 = hashlib.sha256(data).hexdigest()

    def resolve(
        self,
        datasets: list[dict[str, str]],
        protected_access: list[str],
    ) -> dict[str, Any]:
        bindings = []
        permissions = set(protected_access)
        for requested in datasets:
            identifier = requested["id"]
            entry = self.entries.get(identifier)
            if entry is None:
                raise DatasetRegistryError(f"dataset id is not registered: {identifier}")
            if entry["role"] != requested["role"]:
                raise DatasetRegistryError(
                    f"dataset {identifier} role mismatch: contract={requested['role']} "
                    f"registry={entry['role']}"
                )
            if entry["protected"] and entry["role"] not in permissions:
                raise DatasetRegistryError(
                    f"dataset {identifier} requires explicit {entry['role']} access"
                )
            bindings.append(dict(entry))
        bindings.sort(key=lambda item: (item["role"], item["id"]))
        public = [
            {
                "id": item["id"],
                "role": item["role"],
                "identity_sha256": item["identity_sha256"],
                "protected": item["protected"],
            }
            for item in bindings
        ]
        identity = {"datasets": public}
        return {
            "registry_sha256": self.registry_sha256,
            "bindings_sha256": canonical_sha256(identity),
            "bindings": bindings,
            "public_bindings": public,
        }
