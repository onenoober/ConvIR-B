#!/usr/bin/env python3
"""Shared declarative contract for route-ready validation and cloud lifecycle."""

from __future__ import annotations

import hashlib
import json
import math
import re
from statistics import NormalDist
from pathlib import Path
from typing import Any, Callable

import capability_registry

SPEC_SCHEMA_VERSION = 2
LEGACY_SPEC_SCHEMA_VERSION = 1
ASSET_SCHEMA_VERSION = 2
SUPPORTED_ASSET_SCHEMA_VERSIONS = {1, 2}
CONTEXT_SCHEMA_VERSION = 1
GENERIC_RUNNER_RELPATH = "experience_docx/tools/run_route_operation.sh"
RUNTIME_SPEC_DIRECTORY = "experience_docx/route_runtime_specs"
MODEL_CAPABILITY_DIRECTORY = "experience_docx/model_capabilities"
PRECISION_CERTIFICATE_DIRECTORY = "experience_docx/precision_certificates"
RUNTIME_BUNDLE_RELPATHS = (
    GENERIC_RUNNER_RELPATH,
    "experience_docx/tools/route_lifecycle.py",
    "experience_docx/tools/route_program_api.py",
    "experience_docx/tools/route_runtime_contract.py",
    "experience_docx/tools/run_telemetry.py",
    "experience_docx/tools/research_program_contract.py",
    "experience_docx/tools/experiment_spec_compiler.py",
    "experience_docx/tools/scientific_contract.py",
    "experience_docx/tools/capability_registry.py",
)
EVIDENCE_ROLES = {
    "engineering_debug", "development_screening", "confirmation", "sealed_final",
}
RESUME_POLICIES = {"none", "complete_units"}
ALLOWED_EVIDENCE_SUFFIXES = {".json", ".csv", ".md", ".txt"}
SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SAFE_RELPATH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./-]{0,511}$")
SAFE_ENV_KEY = re.compile(r"^CONVIR_ROUTE_[A-Z0-9_]{1,96}$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ASSET_PATH_TOKENS = ("{REMOTE_REPO}", "{RUN_ROOT}", "{OUTPUT_PATH}")
ASSET_ACCESS_ROLES = {
    "unrestricted", "engineering_debug", "development_screening",
    "confirmation", "canary", "sealed_final",
}
ENGINEERING_CONTRACT_MODES = {
    "metadata_only", "cpu_exact", "cpu_reference_equivalent",
    "gpu_synthetic_no_data",
}
ENGINEERING_COST_STRATEGIES = {"same_scale_probe", "fixed_linear_extrapolation"}
ENGINEERING_WORKLOAD_CLASSES = {
    "fixed_iteration_map", "adaptive_search", "variable_graph_or_matrix",
}
PRECISION_MODES = {"formal_precision", "descriptive_capacity", "not_applicable"}
COMPLETE_UNIT_LEDGER_ASSET_ID = "completed_unit_ledger"


class ContractError(ValueError):
    pass


def canonical_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def require_token(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SAFE_TOKEN.fullmatch(value):
        raise ContractError(f"{name} must be a safe token")
    return value


def require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{name} must be boolean")
    return value


def require_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ContractError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def require_number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) \
            or not math.isfinite(float(value)) or not minimum <= float(value) <= maximum:
        raise ContractError(f"{name} must be in [{minimum}, {maximum}]")
    return float(value)


def require_relpath(value: Any, name: str, *, prefix: str | None = None,
                    suffix: str | None = None) -> str:
    if not isinstance(value, str) or not SAFE_RELPATH.fullmatch(value) \
            or value.startswith("/") or ".." in Path(value).parts or "//" in value:
        raise ContractError(f"{name} must be a safe repository-relative path")
    if prefix is not None and not value.startswith(prefix):
        raise ContractError(f"{name} must start with {prefix}")
    if suffix is not None and not value.endswith(suffix):
        raise ContractError(f"{name} must end with {suffix}")
    return value


def safe_join(root: Path, relpath: str) -> Path:
    candidate = (root / relpath).absolute()
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ContractError(f"path escapes root: {relpath}") from exc
    return candidate


def require_asset_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024 \
            or any(ord(character) < 32 for character in value):
        raise ContractError(f"{name} is invalid")
    tokens = [token for token in ASSET_PATH_TOKENS if token in value]
    if any(character in value for character in "{}"):
        if len(tokens) != 1 or not value.startswith(tokens[0]) \
                or any(token in value[len(tokens[0]):] for token in ASSET_PATH_TOKENS):
            raise ContractError(f"{name} has an invalid path token")
        suffix = value[len(tokens[0]):]
        if suffix and not suffix.startswith("/"):
            raise ContractError(f"{name} token must be followed by an absolute suffix")
        if ".." in Path(suffix).parts:
            raise ContractError(f"{name} cannot escape its token root")
    elif not Path(value).is_absolute():
        raise ContractError(f"{name} must be absolute or use one supported root token")
    return value


def runtime_spec_relpath(operation_id: str) -> str:
    return f"{RUNTIME_SPEC_DIRECTORY}/{require_token(operation_id, 'operation_id')}.json"


def _validate_permissions(value: Any) -> dict[str, bool]:
    expected = {"allow_confirmation", "allow_canary", "allow_locked_test"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ContractError("protected_data_permissions has an invalid contract")
    return {key: require_bool(value[key], f"protected_data_permissions.{key}") for key in sorted(expected)}


def _validate_evidence_files(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= 32:
        raise ContractError("evidence_files must contain 1-32 entries")
    result = []
    destinations = set()
    for index, item in enumerate(value):
        name = f"evidence_files[{index}]"
        expected = {"source_relpath", "destination_filename", "required", "max_bytes"}
        if not isinstance(item, dict) or set(item) != expected:
            raise ContractError(f"{name} has an invalid field contract")
        source = require_relpath(item["source_relpath"], f"{name}.source_relpath")
        if not source.startswith("workload/"):
            raise ContractError(f"{name}.source_relpath must be under workload/")
        destination = item["destination_filename"]
        if not isinstance(destination, str) or not SAFE_TOKEN.fullmatch(destination) \
                or Path(destination).name != destination:
            raise ContractError(f"{name}.destination_filename must be a filename")
        if Path(destination).suffix.lower() not in ALLOWED_EVIDENCE_SUFFIXES:
            raise ContractError(f"{name}.destination_filename has a forbidden suffix")
        if destination in destinations:
            raise ContractError("evidence destination filenames must be unique")
        destinations.add(destination)
        result.append({
            "source_relpath": source,
            "destination_filename": destination,
            "required": require_bool(item["required"], f"{name}.required"),
            "max_bytes": require_int(item["max_bytes"], f"{name}.max_bytes", 1, 1024 * 1024),
        })
    return result


def _validate_cost_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError("engineering_contract.cost_contract must be an object")
    strategy = value.get("strategy")
    if strategy not in ENGINEERING_COST_STRATEGIES:
        raise ContractError(
            f"cost_contract.strategy must be one of {sorted(ENGINEERING_COST_STRATEGIES)}"
        )
    common = {"strategy", "workload_class", "formal_iterations", "max_peak_memory_mib"}
    workload_class = value.get("workload_class")
    if workload_class not in ENGINEERING_WORKLOAD_CLASSES:
        raise ContractError("cost_contract.workload_class is invalid")
    normalized = {
        "strategy": strategy,
        "workload_class": workload_class,
        "formal_iterations": require_int(
            value.get("formal_iterations"), "cost_contract.formal_iterations",
            1, 100_000_000,
        ),
        "max_peak_memory_mib": require_int(
            value.get("max_peak_memory_mib"), "cost_contract.max_peak_memory_mib",
            1, 1_048_576,
        ),
    }
    if strategy == "same_scale_probe":
        expected = common | {"max_wall_seconds"}
        if set(value) != expected:
            raise ContractError("same_scale_probe cost contract has an invalid field contract")
        normalized["max_wall_seconds"] = require_int(
            value["max_wall_seconds"], "cost_contract.max_wall_seconds", 1, 7 * 24 * 3600,
        )
        return normalized
    expected = common | {
        "probe_iterations", "max_seconds_per_iteration", "fixed_overhead_seconds",
        "safety_factor", "formal_max_wall_seconds", "memory_scaling",
        "batch_shape_policy", "termination_policy", "candidate_schedule",
        "production_path",
    }
    if set(value) != expected:
        raise ContractError("fixed_linear_extrapolation cost contract has an invalid field contract")
    required_enums = {
        "workload_class": "fixed_iteration_map",
        "memory_scaling": "constant",
        "termination_policy": "fixed_count",
        "candidate_schedule": "none",
        "production_path": "exact",
    }
    if any(value.get(key) != expected_value for key, expected_value in required_enums.items()):
        raise ContractError(
            "fixed linear extrapolation requires a fixed-count exact production map with constant memory"
        )
    if value.get("batch_shape_policy") not in {"fixed", "bounded"}:
        raise ContractError("fixed linear extrapolation requires fixed or bounded batch shapes")
    probe = require_int(
        value["probe_iterations"], "cost_contract.probe_iterations", 1,
        normalized["formal_iterations"],
    )
    if probe >= normalized["formal_iterations"]:
        raise ContractError("fixed linear probe must be smaller than the formal iteration count")
    seconds_per_iteration = require_number(
        value["max_seconds_per_iteration"],
        "cost_contract.max_seconds_per_iteration", 0.000001, 86_400,
    )
    overhead = require_number(
        value["fixed_overhead_seconds"],
        "cost_contract.fixed_overhead_seconds", 0, 86_400,
    )
    safety = require_number(value["safety_factor"], "cost_contract.safety_factor", 1.25, 10)
    formal_bound = require_int(
        value["formal_max_wall_seconds"],
        "cost_contract.formal_max_wall_seconds", 1, 7 * 24 * 3600,
    )
    computed_bound = overhead + safety * normalized["formal_iterations"] * seconds_per_iteration
    if formal_bound + 1e-9 < computed_bound:
        raise ContractError("formal_max_wall_seconds is below the mechanical linear bound")
    normalized.update({
        "probe_iterations": probe,
        "max_seconds_per_iteration": seconds_per_iteration,
        "fixed_overhead_seconds": overhead,
        "safety_factor": safety,
        "formal_max_wall_seconds": formal_bound,
        "memory_scaling": "constant",
        "batch_shape_policy": value["batch_shape_policy"],
        "termination_policy": "fixed_count",
        "candidate_schedule": "none",
        "production_path": "exact",
    })
    return normalized


def _validate_engineering_contract(value: Any, *, legacy: bool) -> dict[str, Any]:
    if legacy:
        return {
            "mode": "cpu_exact", "capability_profile_relpath": None,
            "max_seconds": 300, "cost_contract": None,
            "legacy_implicit_contract": True,
        }
    legacy_expected = {"mode", "capability_profile_relpath", "max_seconds"}
    current_expected = legacy_expected | {"cost_contract"}
    if not isinstance(value, dict) or frozenset(value) not in {
        frozenset(legacy_expected), frozenset(current_expected),
    }:
        raise ContractError("engineering_contract has an invalid field contract")
    mode = value["mode"]
    if mode not in ENGINEERING_CONTRACT_MODES:
        raise ContractError(f"engineering_contract.mode must be one of {sorted(ENGINEERING_CONTRACT_MODES)}")
    profile = value["capability_profile_relpath"]
    if mode == "metadata_only":
        if profile is not None:
            raise ContractError("metadata-only contract cannot declare a capability profile")
    else:
        profile = require_relpath(
            profile, "engineering_contract.capability_profile_relpath",
            prefix=f"{MODEL_CAPABILITY_DIRECTORY}/", suffix=".json",
        )
    return {
        "mode": mode, "capability_profile_relpath": profile,
        "max_seconds": require_int(value["max_seconds"], "engineering_contract.max_seconds", 1, 900),
        "cost_contract": (
            _validate_cost_contract(value["cost_contract"])
            if value.get("cost_contract") is not None else None
        ),
        "legacy_implicit_contract": False,
    }


def _validate_precision_contract(value: Any, *, legacy: bool, role: str,
                                 permissions: dict[str, bool]) -> dict[str, Any]:
    if legacy:
        return {
            "mode": "not_applicable", "certificate_relpath": None,
            "rationale": "legacy runtime schema 1", "legacy_implicit_contract": True,
        }
    expected = {"mode", "certificate_relpath", "rationale"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ContractError("precision_contract has an invalid field contract")
    mode = value["mode"]
    if mode not in PRECISION_MODES:
        raise ContractError(f"precision_contract.mode must be one of {sorted(PRECISION_MODES)}")
    rationale = value["rationale"]
    if not isinstance(rationale, str) or not 8 <= len(rationale) <= 512:
        raise ContractError("precision_contract.rationale must contain 8-512 characters")
    certificate = value["certificate_relpath"]
    if mode == "not_applicable":
        if certificate is not None or role != "engineering_debug":
            raise ContractError("precision not_applicable is restricted to engineering_debug operations")
    else:
        certificate = require_relpath(
            certificate, "precision_contract.certificate_relpath",
            prefix=f"{PRECISION_CERTIFICATE_DIRECTORY}/", suffix=".json",
        )
    if mode == "descriptive_capacity":
        if role != "development_screening" or any(permissions.values()):
            raise ContractError("descriptive capacity requires unprotected development_screening evidence")
    return {
        "mode": mode, "certificate_relpath": certificate, "rationale": rationale,
        "legacy_implicit_contract": False,
    }


def validate_runtime_spec(value: Any, manifest: dict[str, Any], operation_id: str) -> dict[str, Any]:
    legacy_expected = {
        "schema_version", "route_id", "operation_id", "entrypoint_relpath",
        "asset_manifest_relpath", "timeout_seconds", "expected_wall_seconds",
        "total_units", "evidence_role", "resume_policy", "protected_data_permissions",
        "environment", "evidence_files",
    }
    current_expected = legacy_expected | {"engineering_contract", "precision_contract"}
    if not isinstance(value, dict):
        raise ContractError("runtime spec must be an object")
    schema = value.get("schema_version")
    legacy = schema == LEGACY_SPEC_SCHEMA_VERSION
    expected = legacy_expected if legacy else current_expected
    if set(value) != expected:
        raise ContractError("runtime spec has an invalid top-level contract")
    if schema not in {LEGACY_SPEC_SCHEMA_VERSION, SPEC_SCHEMA_VERSION}:
        raise ContractError(f"runtime spec must use schema {LEGACY_SPEC_SCHEMA_VERSION} or {SPEC_SCHEMA_VERSION}")
    if not isinstance(manifest, dict) or operation_id not in manifest.get("operations", {}):
        raise ContractError("runtime spec operation is absent from route manifest")
    operation = manifest["operations"][operation_id]
    route_id = require_token(value["route_id"], "route_id")
    if route_id != manifest.get("route_id"):
        raise ContractError("runtime spec route_id does not match route manifest")
    if require_token(value["operation_id"], "operation_id") != operation_id:
        raise ContractError("runtime spec operation_id mismatch")
    if operation.get("runner_relpath") != GENERIC_RUNNER_RELPATH:
        raise ContractError("route-ready operations must use the generic runner")
    entrypoint = require_relpath(
        value["entrypoint_relpath"], "entrypoint_relpath",
        prefix="experience_docx/tools/", suffix=".py",
    )
    if entrypoint in RUNTIME_BUNDLE_RELPATHS:
        raise ContractError("entrypoint must be route-specific")
    asset_path = value["asset_manifest_relpath"]
    if asset_path is not None:
        asset_path = require_relpath(
            asset_path, "asset_manifest_relpath",
            prefix="experience_docx/route_assets/", suffix=".json",
        )
    timeout = require_int(value["timeout_seconds"], "timeout_seconds", 1, 7 * 24 * 3600)
    expected_wall = require_int(
        value["expected_wall_seconds"], "expected_wall_seconds", 1, timeout,
    )
    role = value["evidence_role"]
    if role not in EVIDENCE_ROLES:
        raise ContractError(f"evidence_role must be one of {sorted(EVIDENCE_ROLES)}")
    resume = value["resume_policy"]
    if resume not in RESUME_POLICIES:
        raise ContractError(f"resume_policy must be one of {sorted(RESUME_POLICIES)}")
    if operation.get("output_policy") != "new":
        raise ContractError("resume_policy and operation output_policy disagree")
    permissions = _validate_permissions(value["protected_data_permissions"])
    if permissions["allow_locked_test"] and role != "sealed_final":
        raise ContractError("locked-test access requires sealed_final evidence role")
    if permissions["allow_confirmation"] and role not in {"confirmation", "sealed_final"}:
        raise ContractError("confirmation access requires confirmation or sealed_final role")
    if resume == "complete_units" and asset_path is None:
        raise ContractError("complete_units recovery requires a typed asset manifest")
    environment = value["environment"]
    if not isinstance(environment, dict) or len(environment) > 32:
        raise ContractError("environment must be an object with at most 32 entries")
    normalized_environment = {}
    for key, item in sorted(environment.items()):
        if not SAFE_ENV_KEY.fullmatch(key):
            raise ContractError(f"environment key is not allowed: {key}")
        if not isinstance(item, str) or len(item) > 512 \
                or any(ord(character) < 32 for character in item):
            raise ContractError(f"environment value is invalid: {key}")
        normalized_environment[key] = item
    engineering = _validate_engineering_contract(
        value.get("engineering_contract"), legacy=legacy,
    )
    if engineering["mode"] == "gpu_synthetic_no_data" and not operation.get("require_gpu"):
        raise ContractError("gpu synthetic contract requires a GPU operation")
    cost = engineering["cost_contract"]
    if cost is not None:
        if engineering["mode"] == "metadata_only":
            raise ContractError("metadata-only engineering cannot declare an iteration cost contract")
        if operation.get("require_gpu") and engineering["mode"] != "gpu_synthetic_no_data":
            raise ContractError("GPU route cost qualification must use gpu_synthetic_no_data")
        if cost["strategy"] == "same_scale_probe":
            if cost["max_wall_seconds"] > engineering["max_seconds"]:
                raise ContractError("same-scale probe exceeds engineering_contract.max_seconds")
        else:
            probe_bound = (
                cost["fixed_overhead_seconds"]
                + cost["probe_iterations"] * cost["max_seconds_per_iteration"]
            )
            if probe_bound > engineering["max_seconds"]:
                raise ContractError("fixed linear probe exceeds engineering_contract.max_seconds")
            if cost["formal_max_wall_seconds"] > expected_wall:
                raise ContractError("fixed linear formal bound exceeds expected_wall_seconds")
        if operation.get("require_gpu") \
                and cost["max_peak_memory_mib"] > operation.get("min_free_gpu_mib", 0):
            raise ContractError("cost memory bound exceeds the frozen GPU free-memory gate")
    precision = _validate_precision_contract(
        value.get("precision_contract"), legacy=legacy, role=role, permissions=permissions,
    )
    return {
        "schema_version": schema,
        "route_id": route_id,
        "operation_id": operation_id,
        "entrypoint_relpath": entrypoint,
        "asset_manifest_relpath": asset_path,
        "timeout_seconds": timeout,
        "expected_wall_seconds": expected_wall,
        "total_units": require_int(value["total_units"], "total_units", 0, 10_000_000),
        "evidence_role": role,
        "resume_policy": resume,
        "protected_data_permissions": permissions,
        "environment": normalized_environment,
        "evidence_files": _validate_evidence_files(value["evidence_files"]),
        "engineering_contract": engineering,
        "precision_contract": precision,
    }


def capability_input_contract_sha256(
    *, contract_mode: str, minimum_fixture: dict[str, int],
    compatibility_imports: list[str], cost_contract: dict[str, Any] | None,
) -> str:
    return canonical_digest({
        "contract_mode": contract_mode,
        "minimum_fixture": minimum_fixture,
        "compatibility_imports": compatibility_imports,
        "cost_contract": cost_contract,
    })


def validate_model_capability(value: Any, spec: dict[str, Any],
                              asset_manifest: dict[str, Any] | None) -> dict[str, Any]:
    expected = {
        "schema_version", "profile_id", "contract_mode", "minimum_fixture",
        "bound_assets", "compatibility_imports", "production_path_statement",
        "protected_data_prohibited", "scientific_output_prohibited",
        "scientific_training_prohibited",
    }
    schema = value.get("schema_version") if isinstance(value, dict) else None
    if schema == 2:
        expected = expected | {"reuse_identity"}
    if not isinstance(value, dict) or set(value) != expected or schema not in {1, 2}:
        raise ContractError("model capability profile has an invalid top-level contract")
    mode = value["contract_mode"]
    if mode != spec["engineering_contract"]["mode"] or mode == "metadata_only":
        raise ContractError("model capability mode does not match the runtime contract")
    fixture = value["minimum_fixture"]
    fixture_fields = {"batch", "channels", "height", "width"}
    if not isinstance(fixture, dict) or set(fixture) != fixture_fields:
        raise ContractError("minimum_fixture has an invalid field contract")
    normalized_fixture = {
        key: require_int(fixture[key], f"minimum_fixture.{key}", 1, 16384)
        for key in sorted(fixture_fields)
    }
    assets = {item["id"]: item for item in (asset_manifest or {}).get("assets", [])}
    bound = value["bound_assets"]
    if not isinstance(bound, list) or not bound:
        raise ContractError("model capability requires at least one bound asset identity")
    normalized_bound = []
    for index, item in enumerate(bound):
        if not isinstance(item, dict) or set(item) != {"id", "identity"}:
            raise ContractError(f"bound_assets[{index}] has an invalid field contract")
        identifier = require_token(item["id"], f"bound_assets[{index}].id")
        asset = assets.get(identifier)
        identity = item["identity"]
        actual = asset.get("sha256") if asset else None
        if actual is None and asset is not None:
            actual = asset.get("commit")
        if asset is None or asset.get("contract_access") is not True:
            raise ContractError(f"bound capability asset is unavailable to contract: {identifier}")
        if not isinstance(identity, str) or not (SHA256.fullmatch(identity) or SHA40.fullmatch(identity)) \
                or actual != identity:
            raise ContractError(f"bound asset identity mismatch: {identifier}")
        normalized_bound.append({"id": identifier, "identity": identity})
    imports = value["compatibility_imports"]
    if not isinstance(imports, list) or len(imports) > 32 \
            or any(not isinstance(item, str) or not item or len(item) > 256 for item in imports):
        raise ContractError("compatibility_imports has an invalid contract")
    statement = value["production_path_statement"]
    if not isinstance(statement, str) or not 16 <= len(statement) <= 1024:
        raise ContractError("production_path_statement must contain 16-1024 characters")
    for key in ("protected_data_prohibited", "scientific_output_prohibited",
                "scientific_training_prohibited"):
        if value[key] is not True:
            raise ContractError(f"model capability requires {key}=true")
    if mode == "gpu_synthetic_no_data":
        exposed = [
            item for item in assets.values()
            if item.get("contract_access")
            and item.get("access_role") not in {"unrestricted", "engineering_debug"}
        ]
        if exposed:
            raise ContractError(
                "gpu synthetic contract can expose only unrestricted/engineering assets"
            )
    result = {
        "schema_version": schema, "profile_id": require_token(value["profile_id"], "profile_id"),
        "contract_mode": mode, "minimum_fixture": normalized_fixture,
        "bound_assets": normalized_bound, "compatibility_imports": imports,
        "production_path_statement": statement,
        "protected_data_prohibited": True, "scientific_output_prohibited": True,
        "scientific_training_prohibited": True,
    }
    if schema == 2:
        try:
            identity = capability_registry.validate_identity(value["reuse_identity"])
        except capability_registry.CapabilityRegistryError as exc:
            raise ContractError(str(exc)) from exc
        bound_identities = {item["identity"] for item in normalized_bound}
        if identity["source_commit"] not in bound_identities:
            raise ContractError("capability source_commit is not bound to an asset")
        for key in (
            "code_path_sha256", "checkpoint_sha256", "runtime_environment_sha256",
        ):
            if identity[key] not in bound_identities:
                raise ContractError(f"capability {key} is not bound to an asset")
        expected_input = capability_input_contract_sha256(
            contract_mode=mode,
            minimum_fixture=normalized_fixture,
            compatibility_imports=imports,
            cost_contract=spec["engineering_contract"].get("cost_contract"),
        )
        if identity["input_contract_sha256"] != expected_input:
            raise ContractError("capability input_contract_sha256 mismatch")
        result["reuse_identity"] = identity
    return result


def _validate_precision_certificate_v2(
    value: Any, spec: dict[str, Any], scientific: dict[str, Any] | None,
) -> dict[str, Any]:
    expected = {
        "schema_version", "certificate_id", "route_id", "operation_id",
        "primary_estimand_id", "independent_unit", "comparison_family",
        "method", "confidence_level", "critical_value", "target_half_width",
        "assurance", "strata", "feasible", "source_role", "source_reference",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ContractError("precision schema 2 has an invalid top-level contract")
    if scientific is None or scientific.get("schema_version") not in {2, 3}:
        raise ContractError("precision schema 2 requires scientific schema 2 or 3")
    if value["route_id"] != spec["route_id"] \
            or value["operation_id"] != spec["operation_id"]:
        raise ContractError("precision certificate identity mismatch")
    estimand = scientific["primary_estimand"]
    uncertainty = scientific["uncertainty"]
    if value["primary_estimand_id"] != estimand["id"]:
        raise ContractError("precision certificate primary estimand mismatch")
    if value["independent_unit"] != estimand["unit"] \
            or value["independent_unit"] != uncertainty["independent_unit"]:
        raise ContractError("precision certificate independent unit mismatch")
    if value["comparison_family"] != uncertainty["comparison_family"]:
        raise ContractError("precision certificate comparison family mismatch")
    confidence = require_number(
        value["confidence_level"], "confidence_level", 0.500000001, 0.999999999,
    )
    if confidence != uncertainty["confidence_level"]:
        raise ContractError("precision certificate confidence level mismatch")
    method = value["method"]
    if method not in {"normal_mean", "binomial_worst_case"}:
        raise ContractError("precision certificate method is invalid")
    critical = require_number(value["critical_value"], "critical_value", 0.01, 20.0)
    family_size = len(scientific["population"]["strata"])
    simultaneous_critical = NormalDist().inv_cdf(
        1.0 - (1.0 - confidence) / (2.0 * family_size)
    )
    if critical + 1e-12 < simultaneous_critical:
        raise ContractError(
            "precision critical_value is below the simultaneous Bonferroni "
            "confidence bound"
        )
    half = require_number(
        value["target_half_width"], "target_half_width", 1e-12, 1e6,
    )
    precision_gates = [
        gate for gate in scientific["gates"] if gate["type"] == "precision"
    ]
    if len(precision_gates) != 1 or any(
        isinstance(gate["threshold"], bool)
        or not isinstance(gate["threshold"], (int, float))
        or float(gate["threshold"]) != half
        for gate in precision_gates
    ):
        raise ContractError("precision target does not match the scientific precision gate")
    assurance = value["assurance"]
    if not isinstance(assurance, dict) or set(assurance) != {
        "method_id", "probability", "planning_sd_rule",
    }:
        raise ContractError("precision assurance has an invalid field contract")
    assurance_value = require_number(
        assurance["probability"], "assurance.probability", 0.500000001, 0.999999999,
    )
    planning_rule = assurance["planning_sd_rule"]
    if not isinstance(planning_rule, str) or not 16 <= len(planning_rule.strip()) <= 1024:
        raise ContractError("assurance.planning_sd_rule must contain 16-1024 characters")
    assurance_value_normalized = {
        "method_id": require_token(assurance["method_id"], "assurance.method_id"),
        "probability": assurance_value,
        "planning_sd_rule": planning_rule.strip(),
    }
    population_strata = {
        item["id"]: item["independent_group_count"]
        for item in scientific["population"]["strata"]
    }
    strata = value["strata"]
    if not isinstance(strata, list) or not strata:
        raise ContractError("precision strata must be a non-empty list")
    normalized_strata = []
    seen = set()
    for index, item in enumerate(strata):
        name = f"strata[{index}]"
        fields = {
            "id", "independent_groups_available", "independent_groups_planned",
            "planning_sd", "planning_sd_upper_bound",
            "independent_groups_required", "feasible", "source_reference",
        }
        if not isinstance(item, dict) or set(item) != fields:
            raise ContractError(f"{name} has an invalid field contract")
        identifier = require_token(item["id"], f"{name}.id")
        if identifier in seen or identifier not in population_strata:
            raise ContractError("precision stratum ids must exactly match the population")
        seen.add(identifier)
        available = require_int(
            item["independent_groups_available"],
            f"{name}.independent_groups_available", 1, 10_000_000,
        )
        if available != population_strata[identifier]:
            raise ContractError(f"{name} available groups differ from the population")
        planned = require_int(
            item["independent_groups_planned"],
            f"{name}.independent_groups_planned", 1, available,
        )
        planning_sd = require_number(item["planning_sd"], f"{name}.planning_sd", 1e-12, 1e6)
        upper_sd = require_number(
            item["planning_sd_upper_bound"], f"{name}.planning_sd_upper_bound",
            planning_sd, 1e6,
        )
        if method == "binomial_worst_case" and (planning_sd != 0.5 or upper_sd != 0.5):
            raise ContractError("binomial_worst_case requires planning SD and upper bound 0.5")
        required = math.ceil((critical * upper_sd / half) ** 2)
        if item["independent_groups_required"] != required:
            raise ContractError(
                f"{name} required groups do not match the frozen upper-bound calculation"
            )
        feasible = planned >= required
        if item["feasible"] is not feasible:
            raise ContractError(f"{name} feasibility flag is inconsistent")
        source = item["source_reference"]
        if not isinstance(source, str) or not 8 <= len(source.strip()) <= 512:
            raise ContractError(f"{name}.source_reference must contain 8-512 characters")
        normalized_strata.append({
            "id": identifier,
            "independent_groups_available": available,
            "independent_groups_planned": planned,
            "planning_sd": planning_sd,
            "planning_sd_upper_bound": upper_sd,
            "independent_groups_required": required,
            "feasible": feasible,
            "source_reference": source.strip(),
        })
    if seen != set(population_strata):
        raise ContractError("precision stratum ids must exactly match the population")
    feasible = all(item["feasible"] for item in normalized_strata)
    if value["feasible"] is not feasible:
        raise ContractError("precision feasibility flag is inconsistent")
    if spec["precision_contract"]["mode"] == "formal_precision" and not feasible:
        raise ContractError("formal precision route is infeasible in at least one stratum")
    role = value["source_role"]
    if role not in {"unrestricted", "engineering_debug", "development_screening"}:
        raise ContractError("precision planning cannot consume protected evidence roles")
    reference = value["source_reference"]
    if not isinstance(reference, str) or not 8 <= len(reference.strip()) <= 512:
        raise ContractError("precision certificate source_reference is invalid")
    return {
        **value,
        "certificate_id": require_token(value["certificate_id"], "certificate_id"),
        "primary_estimand_id": estimand["id"],
        "independent_unit": estimand["unit"],
        "comparison_family": uncertainty["comparison_family"],
        "confidence_level": confidence,
        "critical_value": critical,
        "target_half_width": half,
        "assurance": assurance_value_normalized,
        "strata": normalized_strata,
        "feasible": feasible,
        "source_reference": reference.strip(),
    }


def validate_precision_certificate(
    value: Any, spec: dict[str, Any], scientific: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(value, dict) and value.get("schema_version") == 2:
        return _validate_precision_certificate_v2(value, spec, scientific)
    expected = {
        "schema_version", "certificate_id", "estimand", "method",
        "confidence_level", "target_half_width", "planning_sd",
        "independent_groups_available", "independent_groups_required",
        "feasible", "source_role", "source_reference",
    }
    if not isinstance(value, dict) or set(value) != expected or value["schema_version"] != 1:
        raise ContractError("precision certificate has an invalid top-level contract")
    if value["method"] not in {"normal_mean", "binomial_worst_case"}:
        raise ContractError("precision certificate method is invalid")
    if value["confidence_level"] != 0.95:
        raise ContractError("precision certificate currently requires confidence_level 0.95")
    half = value["target_half_width"]
    sd = value["planning_sd"]
    if not isinstance(half, (int, float)) or isinstance(half, bool) or not 0 < half < 1e6:
        raise ContractError("target_half_width must be positive")
    if value["method"] == "binomial_worst_case":
        if sd != 0.5:
            raise ContractError("binomial_worst_case requires planning_sd 0.5")
    elif not isinstance(sd, (int, float)) or isinstance(sd, bool) or not 0 < sd < 1e6:
        raise ContractError("planning_sd must be positive")
    required = math.ceil((1.959963984540054 * float(sd) / float(half)) ** 2)
    available = require_int(
        value["independent_groups_available"], "independent_groups_available", 1, 10_000_000,
    )
    if value["independent_groups_required"] != required:
        raise ContractError("independent_groups_required does not match the frozen calculation")
    feasible = available >= required
    if value["feasible"] is not feasible:
        raise ContractError("precision feasibility flag is inconsistent")
    mode = spec["precision_contract"]["mode"]
    if mode == "formal_precision" and not feasible:
        raise ContractError("formal precision route is infeasible at the available independent-group count")
    role = value["source_role"]
    if role not in {"unrestricted", "engineering_debug", "development_screening"}:
        raise ContractError("precision planning cannot consume protected evidence roles")
    reference = value["source_reference"]
    estimand = value["estimand"]
    if not isinstance(reference, str) or not 8 <= len(reference) <= 512 \
            or not isinstance(estimand, str) or not 3 <= len(estimand) <= 256:
        raise ContractError("precision certificate text fields are invalid")
    return {**value, "computed_required_groups": required}


def validate_asset_manifest(value: Any, spec: dict[str, Any]) -> dict[str, Any]:
    expected = {"schema_version", "route_id", "operation_id", "assets"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ContractError("asset manifest has an invalid top-level contract")
    asset_schema = value["schema_version"]
    if asset_schema not in SUPPORTED_ASSET_SCHEMA_VERSIONS:
        raise ContractError(
            f"asset manifest must use one of {sorted(SUPPORTED_ASSET_SCHEMA_VERSIONS)}"
        )
    if value["route_id"] != spec["route_id"] or value["operation_id"] != spec["operation_id"]:
        raise ContractError("asset manifest identity mismatch")
    assets = value["assets"]
    if not isinstance(assets, list) or not 1 <= len(assets) <= 128:
        raise ContractError("assets must contain 1-128 entries")
    result = []
    identifiers = set()
    for index, item in enumerate(assets):
        name = f"assets[{index}]"
        if not isinstance(item, dict):
            raise ContractError(f"{name} must be an object")
        kind = item.get("kind")
        common = {"id", "kind", "path", "access_role", "contract_access"}
        required = {
            "file": common | {"sha256"},
            "directory": common,
            "git_checkout": common | {"commit", "require_clean"},
        }
        if kind not in required or set(item) != required[kind]:
            raise ContractError(f"{name} has an invalid {kind!r} contract")
        identifier = require_token(item["id"], f"{name}.id")
        if identifier in identifiers:
            raise ContractError("asset ids must be unique")
        identifiers.add(identifier)
        path = require_asset_path(item["path"], f"{name}.path")
        access_role = item["access_role"]
        if access_role not in ASSET_ACCESS_ROLES:
            raise ContractError(f"{name}.access_role is invalid")
        contract_access = require_bool(item["contract_access"], f"{name}.contract_access")
        if contract_access and access_role in {"confirmation", "canary", "sealed_final"}:
            raise ContractError(f"{name} cannot expose protected data to contract phase")
        permissions = spec["protected_data_permissions"]
        required_permission = {
            "confirmation": "allow_confirmation",
            "canary": "allow_canary",
            "sealed_final": "allow_locked_test",
        }.get(access_role)
        if required_permission is not None and not permissions[required_permission]:
            raise ContractError(f"{name} is not permitted by the runtime spec")
        normalized = {
            "id": identifier, "kind": kind, "path": path,
            "access_role": access_role, "contract_access": contract_access,
        }
        if kind == "file":
            if not isinstance(item["sha256"], str) or not SHA256.fullmatch(item["sha256"]):
                raise ContractError(f"{name}.sha256 is invalid")
            normalized["sha256"] = item["sha256"]
        elif kind == "git_checkout":
            if not isinstance(item["commit"], str) or not SHA40.fullmatch(item["commit"]):
                raise ContractError(f"{name}.commit is invalid")
            normalized.update({
                "commit": item["commit"],
                "require_clean": require_bool(item["require_clean"], f"{name}.require_clean"),
            })
        result.append(normalized)
    if asset_schema == 2 and spec["resume_policy"] == "complete_units":
        ledger = [
            item for item in result if item["id"] == COMPLETE_UNIT_LEDGER_ASSET_ID
        ]
        if len(ledger) != 1 or ledger[0]["kind"] != "file" \
                or ledger[0]["contract_access"] is not False \
                or ledger[0]["access_role"] != "unrestricted":
            raise ContractError(
                "complete_units requires one unrestricted run-only completed_unit_ledger file asset"
            )
    return {
        "schema_version": asset_schema,
        "route_id": spec["route_id"],
        "operation_id": spec["operation_id"],
        "assets": result,
    }


def repository_asset_identity_errors(
    asset_manifest: dict[str, Any], read_repo_file: Callable[[str], bytes],
) -> list[str]:
    """Verify SHA identities for file assets delivered from the route checkout."""
    token = "{REMOTE_REPO}/"
    errors = []
    for index, asset in enumerate(asset_manifest.get("assets", [])):
        path = asset.get("path")
        if asset.get("kind") != "file" or not isinstance(path, str) \
                or not path.startswith(token):
            continue
        relpath = path[len(token):]
        try:
            require_relpath(relpath, f"assets[{index}].path")
            raw = read_repo_file(relpath)
        except Exception as exc:
            errors.append(
                f"assets[{index}] {asset.get('id')!r} cannot read repository file "
                f"{relpath!r}: {exc}"
            )
            continue
        observed = hashlib.sha256(raw).hexdigest()
        declared = asset.get("sha256")
        if declared != observed:
            errors.append(
                f"assets[{index}] {asset.get('id')!r} SHA-256 mismatch for "
                f"{relpath!r}: declared {declared}, observed {observed}"
            )
    return errors


def validate_repository_asset_identities(
    asset_manifest: dict[str, Any], read_repo_file: Callable[[str], bytes],
) -> None:
    errors = repository_asset_identity_errors(asset_manifest, read_repo_file)
    if errors:
        raise ContractError("repository-bound asset identity errors: " + "; ".join(errors))
