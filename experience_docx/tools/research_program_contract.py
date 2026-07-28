#!/usr/bin/env python3
"""Validate flexible, evidence-bound research-program governance contracts.

The contract prevents unlimited adjacent search without making a closed family a
global scientific dead end.  It deliberately separates three route mechanisms:
adjacent, orthogonal, and evidence-backed reopen.  The module validates
authorization; it never chooses a question, model, threshold, or dataset.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable


class ProgramContractError(RuntimeError):
    pass


SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
EVIDENCE_ROLES = {
    "engineering_debug", "development_screening", "confirmation",
    "sealed_final",
}
DEPENDENCY_USES = {
    "baseline", "initialization", "training", "supervision",
    "action_definition", "diagnostic", "evaluation", "comparison",
}
FAMILY_STATES = {"open", "paused", "closed"}
MECHANISM_TYPES = {"adjacent", "orthogonal", "reopen"}
PROTECTED_KEYS = {
    "allow_confirmation", "allow_canary", "allow_locked_test",
}
AMENDMENT_KINDS = {
    "dependency_uses", "adjacent_budget", "attempts_used",
    "family_state", "reopen_evidence_types", "stage_scope",
    "orthogonal_dimensions",
}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ) + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require_object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ProgramContractError(f"{name} must contain exactly {sorted(fields)}")
    return value


def _require_token(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SAFE_TOKEN.fullmatch(value):
        raise ProgramContractError(f"{name} must be a safe token")
    return value


def _require_text(value: Any, name: str, minimum: int = 8, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not minimum <= len(value.strip()) <= maximum:
        raise ProgramContractError(f"{name} must contain {minimum}-{maximum} characters")
    return value.strip()


def _require_int(value: Any, name: str, minimum: int = 0, maximum: int = 1_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ProgramContractError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _require_token_list(value: Any, name: str, *, choices: set[str] | None = None,
                        allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ProgramContractError(f"{name} must be a non-empty list")
    result = [_require_token(item, f"{name}[]") for item in value]
    if len(result) != len(set(result)):
        raise ProgramContractError(f"{name} contains duplicates")
    if choices is not None and not set(result) <= choices:
        raise ProgramContractError(f"{name} contains unsupported values")
    return result


def _require_relpath(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ProgramContractError(f"{name} must be a repository-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not value.startswith("experience_docx/"):
        raise ProgramContractError(f"{name} must stay below experience_docx/")
    return value


def _default_evidence_exists(repo_root: Path) -> Callable[[str], bool]:
    root = repo_root.resolve()

    def exists(relpath: str) -> bool:
        candidate = (root / relpath).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return False
        return candidate.is_file()

    return exists


def _validate_permissions(value: Any, name: str) -> dict[str, bool]:
    result = _require_object(value, PROTECTED_KEYS, name)
    if any(not isinstance(result[key], bool) for key in PROTECTED_KEYS):
        raise ProgramContractError(f"{name} values must be boolean")
    if result["allow_locked_test"] and not result["allow_confirmation"]:
        raise ProgramContractError(f"{name} locked-test access requires confirmation access")
    return dict(result)


def _validate_stage(value: Any, name: str, known_families: set[str]) -> dict[str, Any]:
    stage = _require_object(
        value, {"evidence_roles", "protected_permissions", "route_families"}, name,
    )
    roles = _require_token_list(stage["evidence_roles"], f"{name}.evidence_roles",
                                choices=EVIDENCE_ROLES)
    permissions = _validate_permissions(stage["protected_permissions"],
                                        f"{name}.protected_permissions")
    families = _require_token_list(stage["route_families"], f"{name}.route_families")
    unknown = set(families) - known_families
    if unknown:
        raise ProgramContractError(f"{name} names unknown route families: {sorted(unknown)}")
    if permissions["allow_locked_test"] and "sealed_final" not in roles:
        raise ProgramContractError(f"{name} locked-test access requires sealed_final evidence")
    if permissions["allow_confirmation"] and not ({"confirmation", "sealed_final"} & set(roles)):
        raise ProgramContractError(f"{name} confirmation access lacks a protected evidence role")
    return {
        "evidence_roles": roles, "protected_permissions": permissions,
        "route_families": families,
    }


def _validate_family(value: Any, name: str) -> dict[str, Any]:
    family = _require_object(value, {
        "core_assumption", "adjacent_budget", "attempts_used", "state",
        "close_condition", "reopen_evidence_types",
    }, name)
    budget = _require_int(family["adjacent_budget"], f"{name}.adjacent_budget", 0, 10_000)
    used = _require_int(family["attempts_used"], f"{name}.attempts_used", 0, 10_000)
    if used > budget:
        raise ProgramContractError(f"{name}.attempts_used exceeds adjacent_budget")
    state = family["state"]
    if state not in FAMILY_STATES:
        raise ProgramContractError(f"{name}.state must be one of {sorted(FAMILY_STATES)}")
    return {
        "core_assumption": _require_text(family["core_assumption"], f"{name}.core_assumption", 16),
        "adjacent_budget": budget,
        "attempts_used": used,
        "state": state,
        "close_condition": _require_text(family["close_condition"], f"{name}.close_condition", 16),
        "reopen_evidence_types": _require_token_list(
            family["reopen_evidence_types"], f"{name}.reopen_evidence_types",
        ),
    }


def _validate_base(value: Any) -> dict[str, Any]:
    program = _require_object(value, {
        "schema_version", "program_id", "objective", "source_anchors",
        "dependency_roles", "stages", "route_families",
        "orthogonal_dimensions", "amendments",
    }, "program contract")
    if program["schema_version"] != 1:
        raise ProgramContractError("program contract schema_version must be 1")
    _require_token(program["program_id"], "program_id")
    _require_text(program["objective"], "objective", 16)
    anchors = program["source_anchors"]
    if not isinstance(anchors, list) or not anchors:
        raise ProgramContractError("source_anchors must be a non-empty list")
    anchor_ids = set()
    normalized_anchors = []
    for index, anchor in enumerate(anchors):
        item = _require_object(anchor, {"id", "reference", "allowed_uses"},
                               f"source_anchors[{index}]")
        anchor_id = _require_token(item["id"], f"source_anchors[{index}].id")
        if anchor_id in anchor_ids:
            raise ProgramContractError("source anchor ids must be unique")
        anchor_ids.add(anchor_id)
        normalized_anchors.append({
            "id": anchor_id,
            "reference": _require_text(item["reference"], f"source_anchors[{index}].reference"),
            "allowed_uses": _require_token_list(
                item["allowed_uses"], f"source_anchors[{index}].allowed_uses",
                choices=DEPENDENCY_USES,
            ),
        })
    roles = program["dependency_roles"]
    if not isinstance(roles, dict) or not roles:
        raise ProgramContractError("dependency_roles must be a non-empty object")
    normalized_roles = {}
    for role, uses in roles.items():
        _require_token(role, "dependency role")
        normalized_roles[role] = _require_token_list(
            uses, f"dependency_roles.{role}", choices=DEPENDENCY_USES,
        )
    families = program["route_families"]
    if not isinstance(families, dict) or not families:
        raise ProgramContractError("route_families must be a non-empty object")
    normalized_families = {}
    for family_id, family in families.items():
        _require_token(family_id, "route family id")
        normalized_families[family_id] = _validate_family(family, f"route_families.{family_id}")
    stages = program["stages"]
    if not isinstance(stages, dict) or not stages:
        raise ProgramContractError("stages must be a non-empty object")
    normalized_stages = {}
    for stage_id, stage in stages.items():
        _require_token(stage_id, "stage id")
        normalized_stages[stage_id] = _validate_stage(
            stage, f"stages.{stage_id}", set(normalized_families),
        )
    return {
        "schema_version": 1,
        "program_id": program["program_id"],
        "objective": program["objective"].strip(),
        "source_anchors": normalized_anchors,
        "dependency_roles": normalized_roles,
        "stages": normalized_stages,
        "route_families": normalized_families,
        "orthogonal_dimensions": _require_token_list(
            program["orthogonal_dimensions"], "orthogonal_dimensions",
        ),
        "amendments": program["amendments"],
    }


def _apply_change(effective: dict[str, Any], change: Any, name: str) -> None:
    item = _require_object(change, {"kind", "target", "value"}, name)
    kind = item["kind"]
    if kind not in AMENDMENT_KINDS:
        raise ProgramContractError(f"{name}.kind is unsupported")
    target = _require_token(item["target"], f"{name}.target")
    value = item["value"]
    if kind == "dependency_uses":
        if target not in effective["dependency_roles"]:
            raise ProgramContractError(f"{name} targets an unknown dependency role")
        effective["dependency_roles"][target] = _require_token_list(
            value, f"{name}.value", choices=DEPENDENCY_USES,
        )
    elif kind in {"adjacent_budget", "attempts_used"}:
        if target not in effective["route_families"]:
            raise ProgramContractError(f"{name} targets an unknown route family")
        effective["route_families"][target][kind] = _require_int(
            value, f"{name}.value", 0, 10_000,
        )
    elif kind == "family_state":
        if target not in effective["route_families"] or value not in FAMILY_STATES:
            raise ProgramContractError(f"{name} has an invalid family state target/value")
        effective["route_families"][target]["state"] = value
    elif kind == "reopen_evidence_types":
        if target not in effective["route_families"]:
            raise ProgramContractError(f"{name} targets an unknown route family")
        effective["route_families"][target]["reopen_evidence_types"] = _require_token_list(
            value, f"{name}.value",
        )
    elif kind == "stage_scope":
        if target not in effective["stages"]:
            raise ProgramContractError(f"{name} targets an unknown stage")
        effective["stages"][target] = _validate_stage(
            value, f"{name}.value", set(effective["route_families"]),
        )
    else:
        if target != effective["program_id"]:
            raise ProgramContractError(f"{name} orthogonal-dimension target must be program_id")
        effective["orthogonal_dimensions"] = _require_token_list(
            value, f"{name}.value",
        )


def validate_program_contract(value: Any, *, repo_root: Path | None = None,
                              evidence_exists: Callable[[str], bool] | None = None) -> dict[str, Any]:
    """Return the effective program after applying evidence-bound amendments."""
    effective = _validate_base(value)
    amendments = effective.pop("amendments")
    if not isinstance(amendments, list):
        raise ProgramContractError("amendments must be a list")
    exists = evidence_exists
    if exists is None and repo_root is not None:
        exists = _default_evidence_exists(repo_root)
    if amendments and exists is None:
        raise ProgramContractError(
            "evidence_exists or repo_root is required when amendments are present"
        )
    amendment_ids = set()
    normalized_amendments = []
    for index, amendment in enumerate(amendments):
        name = f"amendments[{index}]"
        item = _require_object(amendment, {
            "id", "reason", "evidence_refs", "approved_scope", "changes",
        }, name)
        amendment_id = _require_token(item["id"], f"{name}.id")
        if amendment_id in amendment_ids:
            raise ProgramContractError("amendment ids must be unique")
        amendment_ids.add(amendment_id)
        refs = item["evidence_refs"]
        if not isinstance(refs, list) or not refs:
            raise ProgramContractError(f"{name}.evidence_refs must be non-empty")
        refs = [_require_relpath(ref, f"{name}.evidence_refs[]") for ref in refs]
        if len(refs) != len(set(refs)):
            raise ProgramContractError(f"{name}.evidence_refs contains duplicates")
        if exists is not None:
            missing = [ref for ref in refs if not exists(ref)]
            if missing:
                raise ProgramContractError(f"{name} evidence is missing: {missing}")
        changes = item["changes"]
        if not isinstance(changes, list) or not changes:
            raise ProgramContractError(f"{name}.changes must be non-empty")
        for change_index, change in enumerate(changes):
            _apply_change(effective, change, f"{name}.changes[{change_index}]")
        normalized_amendments.append({
            "id": amendment_id,
            "reason": _require_text(item["reason"], f"{name}.reason", 16),
            "evidence_refs": refs,
            "approved_scope": _require_text(item["approved_scope"], f"{name}.approved_scope", 8),
            "changes": copy.deepcopy(changes),
        })
    for family_id, family in effective["route_families"].items():
        if family["attempts_used"] > family["adjacent_budget"]:
            raise ProgramContractError(
                f"effective route_families.{family_id}.attempts_used exceeds adjacent_budget"
            )
    effective["amendments"] = normalized_amendments
    effective["contract_sha256"] = canonical_sha256(value)
    return effective


def validate_route_authorization(program: dict[str, Any], claim: Any, *,
                                 repo_root: Path | None = None,
                                 evidence_exists: Callable[[str], bool] | None = None) -> dict[str, Any]:
    """Validate one route claim against an already effective program."""
    route = _require_object(claim, {
        "program_id", "stage_id", "family_id", "mechanism_type",
        "evidence_role", "protected_permissions", "dependencies",
        "adjacent_sequence", "orthogonal_changes", "reopen_evidence",
    }, "route authorization")
    if route["program_id"] != program["program_id"]:
        raise ProgramContractError("route program_id mismatch")
    stage_id = _require_token(route["stage_id"], "stage_id")
    family_id = _require_token(route["family_id"], "family_id")
    if stage_id not in program["stages"] or family_id not in program["route_families"]:
        raise ProgramContractError("route names an unknown stage or family")
    stage = program["stages"][stage_id]
    family = program["route_families"][family_id]
    if family_id not in stage["route_families"]:
        raise ProgramContractError("route family is outside the selected stage")
    evidence_role = route["evidence_role"]
    if evidence_role not in stage["evidence_roles"]:
        raise ProgramContractError("route evidence role is outside the selected stage")
    permissions = _validate_permissions(route["protected_permissions"], "protected_permissions")
    for key in PROTECTED_KEYS:
        if permissions[key] and not stage["protected_permissions"][key]:
            raise ProgramContractError(f"route exceeds stage permission: {key}")
    dependencies = route["dependencies"]
    if not isinstance(dependencies, list):
        raise ProgramContractError("dependencies must be a list")
    normalized_dependencies = []
    for index, dependency in enumerate(dependencies):
        item = _require_object(dependency, {"role", "use", "anchor_id"},
                               f"dependencies[{index}]")
        role = _require_token(item["role"], f"dependencies[{index}].role")
        use = _require_token(item["use"], f"dependencies[{index}].use")
        anchor_id = _require_token(item["anchor_id"], f"dependencies[{index}].anchor_id")
        if role not in program["dependency_roles"] or use not in program["dependency_roles"][role]:
            raise ProgramContractError(f"dependencies[{index}] role does not authorize use {use}")
        anchors = {anchor["id"]: anchor for anchor in program["source_anchors"]}
        if anchor_id not in anchors or use not in anchors[anchor_id]["allowed_uses"]:
            raise ProgramContractError(f"dependencies[{index}] anchor does not authorize use {use}")
        normalized_dependencies.append(dict(item))
    mechanism = route["mechanism_type"]
    if mechanism not in MECHANISM_TYPES:
        raise ProgramContractError(f"mechanism_type must be one of {sorted(MECHANISM_TYPES)}")
    orthogonal = route["orthogonal_changes"]
    reopen = route["reopen_evidence"]
    sequence = route["adjacent_sequence"]
    if mechanism == "adjacent":
        if family["state"] != "open":
            raise ProgramContractError("adjacent route requires an open family")
        expected = family["attempts_used"] + 1
        if _require_int(sequence, "adjacent_sequence", 1, 10_000) != expected:
            raise ProgramContractError(f"adjacent_sequence must equal {expected}")
        if expected > family["adjacent_budget"]:
            raise ProgramContractError("adjacent route budget is exhausted")
        if orthogonal or reopen:
            raise ProgramContractError("adjacent route cannot claim orthogonal/reopen evidence")
    elif mechanism == "orthogonal":
        if sequence is not None or reopen:
            raise ProgramContractError("orthogonal route cannot claim adjacent/reopen fields")
        if not isinstance(orthogonal, list) or not orthogonal:
            raise ProgramContractError("orthogonal route must name a substantive change")
        seen_dimensions = set()
        for index, change in enumerate(orthogonal):
            item = _require_object(change, {"dimension", "reason"},
                                   f"orthogonal_changes[{index}]")
            dimension = _require_token(item["dimension"],
                                       f"orthogonal_changes[{index}].dimension")
            if dimension not in program["orthogonal_dimensions"]:
                raise ProgramContractError(f"orthogonal_changes[{index}] dimension is not allowed")
            if dimension in seen_dimensions:
                raise ProgramContractError("orthogonal change dimensions must be unique")
            seen_dimensions.add(dimension)
            _require_text(item["reason"], f"orthogonal_changes[{index}].reason", 16)
    else:
        if family["state"] != "closed":
            raise ProgramContractError("reopen route requires a closed family")
        if sequence is not None or orthogonal:
            raise ProgramContractError("reopen route cannot claim adjacent/orthogonal fields")
        if not isinstance(reopen, list) or not reopen:
            raise ProgramContractError("reopen route requires new evidence")
        exists = evidence_exists
        if exists is None and repo_root is not None:
            exists = _default_evidence_exists(repo_root)
        for index, evidence in enumerate(reopen):
            item = _require_object(evidence, {"type", "relpath"},
                                   f"reopen_evidence[{index}]")
            evidence_type = _require_token(item["type"], f"reopen_evidence[{index}].type")
            if evidence_type not in family["reopen_evidence_types"]:
                raise ProgramContractError(f"reopen_evidence[{index}] type is not authorized")
            relpath = _require_relpath(item["relpath"], f"reopen_evidence[{index}].relpath")
            if exists is None or not exists(relpath):
                raise ProgramContractError(f"reopen_evidence[{index}] is not verifiably present")
    return {
        "program_id": program["program_id"], "stage_id": stage_id,
        "family_id": family_id, "mechanism_type": mechanism,
        "evidence_role": evidence_role, "protected_permissions": permissions,
        "dependencies": normalized_dependencies,
        "adjacent_budget_consumed": mechanism == "adjacent",
        "authorization_sha256": canonical_sha256(claim),
    }
