#!/usr/bin/env python3
"""Pure validation and evaluation for forward scientific contracts."""

from __future__ import annotations

import itertools
import hashlib
import json
import re
from typing import Any, Callable


class ScientificContractError(ValueError):
    pass


SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
DOI_IDENTIFIER = re.compile(r"^doi:10\.\d{4,9}/\S{2,}$", re.IGNORECASE)
ARXIV_IDENTIFIER = re.compile(
    r"^arxiv:(?:\d{4}\.\d{4,5}|[a-z][a-z.-]*/\d{7})(?:v\d+)?$",
    re.IGNORECASE,
)
OFFICIAL_URL_IDENTIFIER = re.compile(r"^https://[^\s]+$", re.IGNORECASE)
TERMINAL_INDEX_RELPATH = "experience_docx/EXPERIMENT_TERMINAL_INDEX.jsonl"
MAX_TERMINAL_INDEX_BYTES = 1024 * 1024
PROGRAM_SOURCE_PREFIX = "experience_docx/research_programs/"
MAX_BOUND_PROGRAM_BYTES = 128 * 1024
TERMINAL_LABELS = ("pass", "fail", "inconclusive")
FAMILY_EFFECTS = {
    "advance", "stop", "allow_predeclared_evidence", "record_only",
}
GATE_OUTCOMES = {
    "integrity": ("pass", "fail", "invalid"),
    "coverage": ("pass", "fail", "invalid"),
    "precision": ("met", "unmet", "invalid"),
    "materiality": ("favorable", "unfavorable", "indeterminate", "invalid"),
    "safety": ("safe", "unsafe", "indeterminate", "invalid"),
}
DECISION_ROLES = {
    "decisive", "validity_veto", "inconclusive_only", "descriptive",
}
DECISION_POLICIES = {"typed_gate_precedence_v1"}
DECISIVE_OUTCOME_LABELS = {
    "materiality": {
        "favorable": "pass", "unfavorable": "fail",
        "indeterminate": "inconclusive", "invalid": "inconclusive",
    },
    "safety": {
        "safe": "pass", "unsafe": "fail",
        "indeterminate": "inconclusive", "invalid": "inconclusive",
    },
}
BOTTLENECK_CLASSES = {
    "scientific_hypothesis", "measurement", "precision", "data_scope",
    "engineering_capability", "governance",
}
DESIGN_STRATEGIES = {
    "single_factor", "multi_arm", "full_factorial", "fractional_factorial",
    "multi_fidelity", "group_sequential", "other",
}
LITERATURE_SOURCE_STATUSES = {
    "peer_reviewed", "author_formal_version", "official_benchmark",
    "official_protocol",
}
RESEARCH_TRIGGER_TYPES = {"post_terminal", "program_foundation"}
SEQUENTIAL_OUTCOME_ACCESS = {"terminal_only", "predeclared_group_sequential"}


def _token(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SAFE_TOKEN.fullmatch(value):
        raise ScientificContractError(f"{name} must be a safe token")
    return value


def _text(value: Any, name: str, minimum: int = 1, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not minimum <= len(value.strip()) <= maximum:
        raise ScientificContractError(
            f"{name} must contain {minimum}-{maximum} characters"
        )
    return value.strip()


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) \
            or not minimum <= value <= maximum:
        raise ScientificContractError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _number(value: Any, name: str) -> float | int | bool | str:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ScientificContractError(f"{name} must be finite")
        return value
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ScientificContractError(f"{name} has an invalid threshold")


def _object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ScientificContractError(
            f"{name} must contain exactly {sorted(fields)}"
        )
    return value


def _terminal_tuple(value: Any, name: str) -> dict[str, str]:
    item = _object(value, {"state", "decision", "authorizes"}, name)
    return {
        "state": _token(item["state"], f"{name}.state"),
        "decision": _token(item["decision"], f"{name}.decision"),
        "authorizes": _token(item["authorizes"], f"{name}.authorizes"),
    }


def archived_terminal_program_ids(
    read_evidence_file: Callable[[str], bytes],
) -> dict[str, str]:
    """Return terminal-record SHA -> typed program id from one frozen snapshot."""
    try:
        raw_index = read_evidence_file(TERMINAL_INDEX_RELPATH)
    except Exception as exc:
        raise ScientificContractError(
            "research update terminal index is unavailable"
        ) from exc
    if not isinstance(raw_index, bytes) or not raw_index \
            or len(raw_index) > MAX_TERMINAL_INDEX_BYTES:
        raise ScientificContractError(
            "research update terminal index has an invalid byte contract"
        )
    result: dict[str, str] = {}
    for raw_line in raw_index.splitlines():
        if not raw_line.strip():
            continue
        record_sha = hashlib.sha256(raw_line).hexdigest()
        try:
            record = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ScientificContractError(
                "research update terminal index contains invalid JSON"
            ) from exc
        if not isinstance(record, dict) or record.get("schema_version") != 2:
            continue
        bundle = record.get("contract_bundle")
        if not isinstance(bundle, list):
            raise ScientificContractError(
                "typed terminal launch bundle is unavailable for program identity"
            )
        matches = [
            item for item in bundle
            if isinstance(item, dict)
            and isinstance(item.get("source_path"), str)
            and item["source_path"].startswith(PROGRAM_SOURCE_PREFIX)
            and item["source_path"].endswith(".json")
        ]
        if not matches:
            continue
        if len(matches) != 1:
            raise ScientificContractError(
                "typed terminal launch bundle has ambiguous program identity"
            )
        binding = matches[0]
        path = binding.get("path")
        size = binding.get("bytes")
        digest = binding.get("sha256")
        if not isinstance(path, str) or not path.startswith(
                "experience_docx/experiment_logs/") \
                or ".." in path.split("/") \
                or isinstance(size, bool) or not isinstance(size, int) \
                or not 1 <= size <= MAX_BOUND_PROGRAM_BYTES \
                or not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise ScientificContractError(
                "typed terminal program binding has an invalid identity contract"
            )
        try:
            raw_program = read_evidence_file(path)
        except Exception as exc:
            raise ScientificContractError(
                "typed terminal program contract is unavailable"
            ) from exc
        if not isinstance(raw_program, bytes) or len(raw_program) != size \
                or hashlib.sha256(raw_program).hexdigest() != digest:
            raise ScientificContractError(
                "typed terminal program contract identity differs"
            )
        try:
            program = json.loads(raw_program)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ScientificContractError(
                "typed terminal program contract is invalid JSON"
            ) from exc
        if not isinstance(program, dict):
            raise ScientificContractError(
                "typed terminal program contract is not an object"
            )
        result[record_sha] = _token(
            program.get("program_id"), "typed terminal program_id",
        )
    return result


def validate_decision_design(
    value: Any, *, strategy: str, hypothesis_ids: set[str],
) -> dict[str, Any]:
    item = _object(value, {
        "arms", "factors", "estimable_terms", "alias_structure",
        "mechanism_estimands", "multiplicity_control", "sequential_plan",
    }, "research_update_binding.decision_design")

    arms = item["arms"]
    if not isinstance(arms, list) or not 2 <= len(arms) <= 16:
        raise ScientificContractError(
            "research update decision design arms must contain 2-16 entries"
        )
    normalized_arms = []
    seen_arms = set()
    for index, arm in enumerate(arms):
        name = f"research_update_binding.decision_design.arms[{index}]"
        arm = _object(arm, {"id", "role", "description"}, name)
        identifier = _token(arm["id"], f"{name}.id")
        if identifier in seen_arms:
            raise ScientificContractError("research update arm ids must be unique")
        seen_arms.add(identifier)
        normalized_arms.append({
            "id": identifier,
            "role": _token(arm["role"], f"{name}.role"),
            "description": _text(arm["description"], f"{name}.description", 8, 1024),
        })

    factors = item["factors"]
    if not isinstance(factors, list) or len(factors) > 8:
        raise ScientificContractError(
            "research update decision design factors must contain 0-8 entries"
        )
    normalized_factors = []
    seen_factors = set()
    for index, factor in enumerate(factors):
        name = f"research_update_binding.decision_design.factors[{index}]"
        factor = _object(factor, {"id", "levels"}, name)
        identifier = _token(factor["id"], f"{name}.id")
        if identifier in seen_factors:
            raise ScientificContractError("research update factor ids must be unique")
        seen_factors.add(identifier)
        levels = factor["levels"]
        if not isinstance(levels, list) or not 2 <= len(levels) <= 16:
            raise ScientificContractError(f"{name}.levels must contain 2-16 entries")
        normalized_levels = [_token(level, f"{name}.levels[]") for level in levels]
        if len(normalized_levels) != len(set(normalized_levels)):
            raise ScientificContractError(f"{name}.levels must be unique")
        normalized_factors.append({"id": identifier, "levels": normalized_levels})

    terms = item["estimable_terms"]
    if not isinstance(terms, list) or not 1 <= len(terms) <= 32:
        raise ScientificContractError(
            "research update decision design estimable_terms must contain 1-32 entries"
        )
    normalized_terms = [
        _text(term, "research_update_binding.decision_design.estimable_terms[]", 1, 256)
        for term in terms
    ]
    if len(normalized_terms) != len(set(normalized_terms)):
        raise ScientificContractError("research update estimable_terms must be unique")

    mechanism_estimands = item["mechanism_estimands"]
    if (
        not isinstance(mechanism_estimands, list)
        or not 1 <= len(mechanism_estimands) <= 8
    ):
        raise ScientificContractError(
            "research update mechanism_estimands must contain 1-8 entries"
        )
    normalized_mechanisms = []
    seen_mechanisms = set()
    for index, estimand in enumerate(mechanism_estimands):
        name = (
            f"research_update_binding.decision_design.mechanism_estimands[{index}]"
        )
        estimand = _object(estimand, {
            "id", "hypothesis_id", "metric_id", "prediction", "falsifier",
        }, name)
        identifier = _token(estimand["id"], f"{name}.id")
        if identifier in seen_mechanisms:
            raise ScientificContractError(
                "research update mechanism estimand ids must be unique"
            )
        seen_mechanisms.add(identifier)
        hypothesis_id = _token(estimand["hypothesis_id"], f"{name}.hypothesis_id")
        if hypothesis_id not in hypothesis_ids:
            raise ScientificContractError(
                f"{name}.hypothesis_id is not a declared live hypothesis"
            )
        normalized_mechanisms.append({
            "id": identifier,
            "hypothesis_id": hypothesis_id,
            "metric_id": _token(estimand["metric_id"], f"{name}.metric_id"),
            "prediction": _text(
                estimand["prediction"], f"{name}.prediction", 16, 2048,
            ),
            "falsifier": _text(
                estimand["falsifier"], f"{name}.falsifier", 16, 2048,
            ),
        })
    covered_hypotheses = {
        estimand["hypothesis_id"] for estimand in normalized_mechanisms
    }
    if covered_hypotheses != hypothesis_ids:
        raise ScientificContractError(
            "every live hypothesis must bind at least one mechanism estimand"
        )

    sequential = _object(item["sequential_plan"], {
        "looks", "alpha_spending", "boundaries", "outcome_access",
    }, "research_update_binding.decision_design.sequential_plan")
    looks = sequential["looks"]
    if not isinstance(looks, list) or not 1 <= len(looks) <= 16:
        raise ScientificContractError(
            "research update sequential looks must contain 1-16 entries"
        )
    normalized_looks = []
    seen_looks = set()
    previous_fraction = 0.0
    for index, look in enumerate(looks):
        name = f"research_update_binding.decision_design.sequential_plan.looks[{index}]"
        look = _object(look, {"id", "information_fraction"}, name)
        identifier = _token(look["id"], f"{name}.id")
        fraction = look["information_fraction"]
        if identifier in seen_looks:
            raise ScientificContractError("research update sequential look ids must be unique")
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not previous_fraction < float(fraction) <= 1.0
        ):
            raise ScientificContractError(
                "research update information fractions must strictly increase in (0, 1]"
            )
        seen_looks.add(identifier)
        previous_fraction = float(fraction)
        normalized_looks.append({
            "id": identifier, "information_fraction": float(fraction),
        })
    if previous_fraction != 1.0:
        raise ScientificContractError(
            "research update sequential plan must end at information_fraction 1.0"
        )
    alpha = _object(sequential["alpha_spending"], {
        "method", "familywise_alpha",
    }, "research_update_binding.decision_design.sequential_plan.alpha_spending")
    familywise_alpha = alpha["familywise_alpha"]
    if (
        isinstance(familywise_alpha, bool)
        or not isinstance(familywise_alpha, (int, float))
        or not 0.0 < float(familywise_alpha) < 0.5
    ):
        raise ScientificContractError(
            "research update familywise_alpha must be in (0, 0.5)"
        )
    alpha_method = _token(
        alpha["method"],
        "research_update_binding.decision_design.sequential_plan.alpha_spending.method",
    )
    boundaries = _object(sequential["boundaries"], {
        "success", "futility", "precision", "validity",
    }, "research_update_binding.decision_design.sequential_plan.boundaries")
    normalized_boundaries = {
        key: _text(
            boundaries[key],
            f"research_update_binding.decision_design.sequential_plan.boundaries.{key}",
            8, 2048,
        )
        for key in ("success", "futility", "precision", "validity")
    }
    outcome_access = sequential["outcome_access"]
    if outcome_access not in SEQUENTIAL_OUTCOME_ACCESS:
        raise ScientificContractError(
            "research update sequential outcome_access is invalid"
        )
    if strategy == "group_sequential":
        if (
            len(normalized_looks) < 2
            or outcome_access != "predeclared_group_sequential"
            or alpha_method == "not_applicable"
        ):
            raise ScientificContractError(
                "group_sequential requires multiple looks, predeclared outcome access, "
                "and an alpha-spending method"
            )
    elif len(normalized_looks) != 1 or outcome_access != "terminal_only":
        raise ScientificContractError(
            "non-sequential strategies permit one terminal-only outcome look"
        )

    if strategy == "single_factor" and len(normalized_factors) != 1:
        raise ScientificContractError("single_factor requires exactly one factor")
    if (
        strategy in {"full_factorial", "fractional_factorial"}
        and len(normalized_factors) < 2
    ):
        raise ScientificContractError(f"{strategy} requires at least two factors")
    alias_structure = _text(
        item["alias_structure"],
        "research_update_binding.decision_design.alias_structure", 4, 2048,
    )
    if (
        strategy == "fractional_factorial"
        and alias_structure.casefold() == "not_applicable"
    ):
        raise ScientificContractError(
            "fractional_factorial requires an explicit alias structure"
        )
    multiplicity = _text(
        item["multiplicity_control"],
        "research_update_binding.decision_design.multiplicity_control", 8, 2048,
    )
    if (
        (
            len(normalized_arms) > 2
            or len(normalized_terms) > 1
            or len(normalized_mechanisms) > 1
        )
        and multiplicity.casefold() in {"none", "not_applicable"}
    ):
        raise ScientificContractError(
            "multi-comparison designs require explicit multiplicity control"
        )
    return {
        "arms": normalized_arms,
        "factors": normalized_factors,
        "estimable_terms": normalized_terms,
        "alias_structure": alias_structure,
        "mechanism_estimands": normalized_mechanisms,
        "multiplicity_control": multiplicity,
        "sequential_plan": {
            "looks": normalized_looks,
            "alpha_spending": {
                "method": alpha_method,
                "familywise_alpha": float(familywise_alpha),
            },
            "boundaries": normalized_boundaries,
            "outcome_access": outcome_access,
        },
    }


def _validate_research_update_binding(
    value: Any, *, expected_snapshot_commit: str | None = None,
    read_evidence_file: Callable[[str], bytes] | None = None,
    require_current_design: bool = False,
) -> dict[str, Any]:
    required_fields = {
        "snapshot_commit", "trigger_type", "trigger_terminals", "bottleneck_class",
        "bottleneck_statement", "literature_basis", "hypotheses",
        "design_selection",
    }
    if (
        not isinstance(value, dict)
        or not required_fields <= set(value)
        or set(value) - (required_fields | {"decision_design"})
    ):
        raise ScientificContractError(
            "research_update_binding has an invalid field contract"
        )
    if require_current_design and "decision_design" not in value:
        raise ScientificContractError(
            "new schema-3 authoring requires research_update_binding.decision_design"
        )
    item = value
    snapshot_commit = item["snapshot_commit"]
    if not isinstance(snapshot_commit, str) or not SHA40.fullmatch(snapshot_commit):
        raise ScientificContractError(
            "research_update_binding.snapshot_commit must be a 40-character Git SHA"
        )
    if expected_snapshot_commit is not None and snapshot_commit != expected_snapshot_commit:
        raise ScientificContractError(
            "research_update_binding.snapshot_commit does not equal authoritative main"
        )

    trigger_type = item["trigger_type"]
    if trigger_type not in RESEARCH_TRIGGER_TYPES:
        raise ScientificContractError(
            "research_update_binding.trigger_type is invalid"
        )
    triggers = item["trigger_terminals"]
    expected_range = (
        isinstance(triggers, list)
        and (
            (trigger_type == "post_terminal" and 1 <= len(triggers) <= 8)
            or (trigger_type == "program_foundation" and len(triggers) == 0)
        )
    )
    if not expected_range:
        raise ScientificContractError(
            "research_update_binding.trigger_terminals must contain 1-8 entries "
            "for post_terminal or zero entries for program_foundation"
        )
    terminal_records: dict[str, dict[str, Any]] = {}
    if triggers and read_evidence_file is not None:
        try:
            raw_index = read_evidence_file(TERMINAL_INDEX_RELPATH)
        except Exception as exc:
            raise ScientificContractError(
                "research update terminal index is unavailable"
            ) from exc
        if not isinstance(raw_index, bytes) or not raw_index \
                or len(raw_index) > MAX_TERMINAL_INDEX_BYTES:
            raise ScientificContractError(
                "research update terminal index has an invalid byte contract"
            )
        for raw_line in raw_index.splitlines():
            if not raw_line.strip():
                continue
            record_sha = hashlib.sha256(raw_line).hexdigest()
            try:
                record = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ScientificContractError(
                    "research update terminal index contains invalid JSON"
                ) from exc
            if record_sha in terminal_records:
                raise ScientificContractError(
                    "research update terminal index contains duplicate record identity"
                )
            terminal_records[record_sha] = record

    normalized_triggers = []
    seen_triggers = set()
    for index, trigger in enumerate(triggers):
        name = f"research_update_binding.trigger_terminals[{index}]"
        trigger = _object(trigger, {"route_id", "terminal_record_sha256"}, name)
        route_id = _token(trigger["route_id"], f"{name}.route_id")
        record_sha = trigger["terminal_record_sha256"]
        if not isinstance(record_sha, str) or not SHA256.fullmatch(record_sha):
            raise ScientificContractError(
                f"{name}.terminal_record_sha256 must be a SHA-256"
            )
        identity = (route_id, record_sha)
        if identity in seen_triggers:
            raise ScientificContractError("research update trigger terminals must be unique")
        seen_triggers.add(identity)
        if read_evidence_file is not None:
            record = terminal_records.get(record_sha)
            if not isinstance(record, dict) or record.get("route_id") != route_id:
                raise ScientificContractError(
                    f"{name} does not bind a matching authoritative terminal record"
                )
        normalized_triggers.append({
            "route_id": route_id, "terminal_record_sha256": record_sha,
        })

    bottleneck_class = item["bottleneck_class"]
    if bottleneck_class not in BOTTLENECK_CLASSES:
        raise ScientificContractError(
            "research_update_binding.bottleneck_class is invalid"
        )
    literature = item["literature_basis"]
    if not isinstance(literature, list) or not 1 <= len(literature) <= 16:
        raise ScientificContractError(
            "research_update_binding.literature_basis must contain 1-16 entries"
        )
    normalized_literature = []
    seen_literature = set()
    for index, source in enumerate(literature):
        name = f"research_update_binding.literature_basis[{index}]"
        source = _object(source, {
            "identifier", "source_status", "task", "transferable_claim",
            "applicability_limit",
        }, name)
        identifier = _text(source["identifier"], f"{name}.identifier", 4, 256)
        if not any(pattern.fullmatch(identifier) for pattern in (
            DOI_IDENTIFIER, ARXIV_IDENTIFIER, OFFICIAL_URL_IDENTIFIER,
        )):
            raise ScientificContractError(
                f"{name}.identifier must be a DOI, arXiv id, or official HTTPS source"
            )
        normalized_identifier = identifier.casefold()
        if normalized_identifier in seen_literature:
            raise ScientificContractError(
                "research update literature identifiers must be unique"
            )
        seen_literature.add(normalized_identifier)
        if source["source_status"] not in LITERATURE_SOURCE_STATUSES:
            raise ScientificContractError(f"{name}.source_status is invalid")
        normalized_literature.append({
            "identifier": identifier,
            "source_status": source["source_status"],
            "task": _text(source["task"], f"{name}.task", 4, 512),
            "transferable_claim": _text(
                source["transferable_claim"], f"{name}.transferable_claim", 16, 2048,
            ),
            "applicability_limit": _text(
                source["applicability_limit"], f"{name}.applicability_limit", 16, 2048,
            ),
        })
    hypotheses = item["hypotheses"]
    if not isinstance(hypotheses, list) or not 2 <= len(hypotheses) <= 8:
        raise ScientificContractError(
            "research_update_binding.hypotheses must contain 2-8 entries"
        )
    normalized_hypotheses = []
    seen_hypotheses = set()
    for index, hypothesis in enumerate(hypotheses):
        name = f"research_update_binding.hypotheses[{index}]"
        hypothesis = _object(hypothesis, {
            "id", "statement", "discriminating_prediction", "falsifier",
        }, name)
        identifier = _token(hypothesis["id"], f"{name}.id")
        if identifier in seen_hypotheses:
            raise ScientificContractError("research update hypothesis ids must be unique")
        seen_hypotheses.add(identifier)
        normalized_hypotheses.append({
            "id": identifier,
            "statement": _text(hypothesis["statement"], f"{name}.statement", 16, 2048),
            "discriminating_prediction": _text(
                hypothesis["discriminating_prediction"],
                f"{name}.discriminating_prediction", 16, 2048,
            ),
            "falsifier": _text(hypothesis["falsifier"], f"{name}.falsifier", 16, 2048),
        })

    design = _object(item["design_selection"], {
        "strategy", "decision_value", "expected_time_to_decision",
        "shared_setup", "worst_case_stopping_cost",
    }, "research_update_binding.design_selection")
    if design["strategy"] not in DESIGN_STRATEGIES:
        raise ScientificContractError(
            "research_update_binding.design_selection.strategy is invalid"
        )
    normalized_design = {"strategy": design["strategy"]}
    for field in (
        "decision_value", "expected_time_to_decision", "shared_setup",
        "worst_case_stopping_cost",
    ):
        normalized_design[field] = _text(
            design[field], f"research_update_binding.design_selection.{field}", 8, 2048,
        )
    result = {
        "snapshot_commit": snapshot_commit,
        "trigger_type": trigger_type,
        "trigger_terminals": normalized_triggers,
        "bottleneck_class": bottleneck_class,
        "bottleneck_statement": _text(
            item["bottleneck_statement"],
            "research_update_binding.bottleneck_statement", 16, 4096,
        ),
        "literature_basis": normalized_literature,
        "hypotheses": normalized_hypotheses,
        "design_selection": normalized_design,
    }
    if "decision_design" in item:
        result["decision_design"] = validate_decision_design(
            item["decision_design"],
            strategy=design["strategy"],
            hypothesis_ids=seen_hypotheses,
        )
    return result


def _validate_population(value: Any) -> dict[str, Any]:
    item = _object(value, {
        "evidence_role", "grouping_unit", "independent_group_count", "strata",
        "allow_confirmation", "allow_canary", "allow_locked_test",
    }, "population")
    role = item["evidence_role"]
    if role not in {
        "engineering_debug", "development_screening", "confirmation", "sealed_final",
    }:
        raise ScientificContractError("population.evidence_role is invalid")
    grouping = _token(item["grouping_unit"], "population.grouping_unit")
    total = _integer(
        item["independent_group_count"], "population.independent_group_count",
        0, 10_000_000,
    )
    strata = item["strata"]
    if not isinstance(strata, list) or not 1 <= len(strata) <= 64:
        raise ScientificContractError("population.strata must contain 1-64 entries")
    normalized_strata = []
    seen = set()
    for index, stratum in enumerate(strata):
        name = f"population.strata[{index}]"
        stratum = _object(stratum, {"id", "independent_group_count"}, name)
        identifier = _token(stratum["id"], f"{name}.id")
        if identifier in seen:
            raise ScientificContractError("population stratum ids must be unique")
        seen.add(identifier)
        normalized_strata.append({
            "id": identifier,
            "independent_group_count": _integer(
                stratum["independent_group_count"],
                f"{name}.independent_group_count", 0, 10_000_000,
            ),
        })
    if sum(item["independent_group_count"] for item in normalized_strata) != total:
        raise ScientificContractError(
            "population stratum counts must sum to independent_group_count"
        )
    permissions = {
        key: item[key]
        for key in ("allow_confirmation", "allow_canary", "allow_locked_test")
    }
    if any(not isinstance(value, bool) for value in permissions.values()):
        raise ScientificContractError("population permissions must be boolean")
    if permissions["allow_locked_test"] and role != "sealed_final":
        raise ScientificContractError("locked test requires sealed_final evidence")
    if permissions["allow_confirmation"] and role not in {"confirmation", "sealed_final"}:
        raise ScientificContractError("confirmation access requires protected evidence")
    return {
        "evidence_role": role,
        "grouping_unit": grouping,
        "independent_group_count": total,
        "strata": normalized_strata,
        **permissions,
    }


def _validate_estimand(value: Any, population: dict[str, Any]) -> dict[str, Any]:
    item = _object(value, {
        "id", "metric_id", "direction", "aggregation", "unit", "strata",
    }, "primary_estimand")
    direction = item["direction"]
    if direction not in {"higher", "lower"}:
        raise ScientificContractError("primary_estimand.direction is invalid")
    strata = item["strata"]
    if not isinstance(strata, list) or not strata:
        raise ScientificContractError("primary_estimand.strata must be non-empty")
    normalized_strata = [_token(value, "primary_estimand.strata[]") for value in strata]
    if len(normalized_strata) != len(set(normalized_strata)):
        raise ScientificContractError("primary_estimand.strata contains duplicates")
    population_strata = {item["id"] for item in population["strata"]}
    if set(normalized_strata) != population_strata:
        raise ScientificContractError(
            "primary_estimand.strata must equal the population strata"
        )
    unit = _token(item["unit"], "primary_estimand.unit")
    if unit != population["grouping_unit"]:
        raise ScientificContractError(
            "primary_estimand.unit must equal population.grouping_unit"
        )
    return {
        "id": _token(item["id"], "primary_estimand.id"),
        "metric_id": _token(item["metric_id"], "primary_estimand.metric_id"),
        "direction": direction,
        "aggregation": _text(
            item["aggregation"], "primary_estimand.aggregation", 8, 2048,
        ),
        "unit": unit,
        "strata": normalized_strata,
    }


def _validate_uncertainty(value: Any, population: dict[str, Any]) -> dict[str, Any]:
    item = _object(value, {
        "id", "method_id", "confidence_level", "independent_unit",
        "comparison_family",
    }, "uncertainty")
    confidence = item["confidence_level"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) \
            or not 0.5 < confidence < 1.0:
        raise ScientificContractError("uncertainty.confidence_level must be in (0.5, 1)")
    unit = _token(item["independent_unit"], "uncertainty.independent_unit")
    if unit != population["grouping_unit"]:
        raise ScientificContractError(
            "uncertainty.independent_unit must equal population.grouping_unit"
        )
    return {
        "id": _token(item["id"], "uncertainty.id"),
        "method_id": _token(item["method_id"], "uncertainty.method_id"),
        "confidence_level": float(confidence),
        "independent_unit": unit,
        "comparison_family": _token(
            item["comparison_family"], "uncertainty.comparison_family",
        ),
    }


def _validate_gates(value: Any, uncertainty: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise ScientificContractError("gates must contain 1-8 entries")
    result = []
    seen = set()
    for index, gate in enumerate(value):
        name = f"gates[{index}]"
        gate = _object(gate, {
            "id", "type", "estimand_id", "reference", "direction", "threshold",
            "uncertainty_id", "comparison_family", "decision_role", "outcomes",
            "neutral_outcome",
        }, name)
        identifier = _token(gate["id"], f"{name}.id")
        if identifier in seen:
            raise ScientificContractError("gate ids must be unique")
        seen.add(identifier)
        gate_type = gate["type"]
        if gate_type not in GATE_OUTCOMES:
            raise ScientificContractError(f"{name}.type is invalid")
        outcomes = gate["outcomes"]
        if not isinstance(outcomes, list) or not outcomes:
            raise ScientificContractError(f"{name}.outcomes must be non-empty")
        normalized_outcomes = [_token(item, f"{name}.outcomes[]") for item in outcomes]
        if len(normalized_outcomes) != len(set(normalized_outcomes)) \
                or set(normalized_outcomes) != set(GATE_OUTCOMES[gate_type]):
            raise ScientificContractError(
                f"{name}.outcomes must equal {list(GATE_OUTCOMES[gate_type])}"
            )
        neutral = _token(gate["neutral_outcome"], f"{name}.neutral_outcome")
        if neutral not in normalized_outcomes:
            raise ScientificContractError(f"{name}.neutral_outcome is not an outcome")
        required_neutral = {
            "integrity": "pass",
            "coverage": "pass",
            "precision": "met",
        }.get(gate_type)
        if required_neutral is not None and neutral != required_neutral:
            raise ScientificContractError(
                f"{name}.neutral_outcome must equal {required_neutral}"
            )
        role = gate["decision_role"]
        if role not in DECISION_ROLES:
            raise ScientificContractError(f"{name}.decision_role is invalid")
        if gate_type in {"integrity", "coverage"} and role != "validity_veto":
            raise ScientificContractError(
                f"{name}.{gate_type} must use validity_veto"
            )
        if gate_type == "precision" and role not in {
            "inconclusive_only", "descriptive",
        }:
            raise ScientificContractError(
                f"{name}.precision cannot be decisive"
            )
        direction = gate["direction"]
        if direction not in {"min", "max", "equal"}:
            raise ScientificContractError(f"{name}.direction is invalid")
        uncertainty_id = _token(gate["uncertainty_id"], f"{name}.uncertainty_id")
        if uncertainty_id != uncertainty["id"]:
            raise ScientificContractError(f"{name}.uncertainty_id mismatch")
        comparison_family = _token(
            gate["comparison_family"], f"{name}.comparison_family",
        )
        if comparison_family != uncertainty["comparison_family"]:
            raise ScientificContractError(f"{name}.comparison_family mismatch")
        result.append({
            "id": identifier,
            "type": gate_type,
            "estimand_id": _token(gate["estimand_id"], f"{name}.estimand_id"),
            "reference": _text(gate["reference"], f"{name}.reference", 8, 1024),
            "direction": direction,
            "threshold": _number(gate["threshold"], f"{name}.threshold"),
            "uncertainty_id": uncertainty_id,
            "comparison_family": comparison_family,
            "decision_role": role,
            "outcomes": normalized_outcomes,
            "neutral_outcome": neutral,
        })
    return result


def _validate_terminal_actions(value: Any, evidence_role: str) -> dict[str, dict[str, Any]]:
    actions = _object(value, set(TERMINAL_LABELS), "terminal_actions")
    normalized = {}
    for label in TERMINAL_LABELS:
        name = f"terminal_actions.{label}"
        item = _object(actions[label], {
            "terminal", "next_action_id", "family_effect",
        }, name)
        terminal = _terminal_tuple(item["terminal"], f"{name}.terminal")
        if evidence_role in {"engineering_debug", "development_screening"} \
                and terminal["authorizes"] in {
                    "PROMOTION", "DEPLOYMENT", "LOCKED_TEST", "SEALED_FINAL",
                }:
            raise ScientificContractError(
                "development evidence cannot authorize promotion or final use"
            )
        next_action = item["next_action_id"]
        if next_action is not None:
            next_action = _token(next_action, f"{name}.next_action_id")
        family_effect = item["family_effect"]
        if family_effect not in FAMILY_EFFECTS:
            raise ScientificContractError(f"{name}.family_effect is invalid")
        allowed_effects = {
            "pass": {"advance", "allow_predeclared_evidence", "record_only"},
            "fail": {"stop", "record_only"},
            "inconclusive": {"stop", "allow_predeclared_evidence", "record_only"},
        }[label]
        if family_effect not in allowed_effects:
            raise ScientificContractError(
                f"{name}.family_effect is inconsistent with {label}"
            )
        if family_effect in {"advance", "allow_predeclared_evidence"} \
                and next_action is None:
            raise ScientificContractError(
                f"{name}.next_action_id is required by family_effect"
            )
        normalized[label] = {
            "terminal": terminal,
            "next_action_id": next_action,
            "family_effect": family_effect,
        }
    terminal_keys = {
        json.dumps(item["terminal"], sort_keys=True, separators=(",", ":"))
        for item in normalized.values()
    }
    if len(terminal_keys) != len(TERMINAL_LABELS):
        raise ScientificContractError("terminal action tuples must be distinct")
    decision_values = {
        (
            item["terminal"]["authorizes"], item["next_action_id"],
            item["family_effect"],
        )
        for item in normalized.values()
    }
    if len(decision_values) != len(TERMINAL_LABELS):
        raise ScientificContractError(
            "each terminal outcome must change authorization, next action, or family effect"
        )
    return normalized


def _normalize_rules(value: Any, gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= 64:
        raise ScientificContractError("decision_table.rules must contain 1-64 entries")
    gate_outcomes = {gate["id"]: set(gate["outcomes"]) for gate in gates}
    result = []
    seen = set()
    for index, rule in enumerate(value):
        name = f"decision_table.rules[{index}]"
        rule = _object(rule, {"id", "when", "terminal"}, name)
        identifier = _token(rule["id"], f"{name}.id")
        if identifier in seen:
            raise ScientificContractError("decision rule ids must be unique")
        seen.add(identifier)
        if rule["terminal"] not in TERMINAL_LABELS:
            raise ScientificContractError(f"{name}.terminal is invalid")
        conditions = rule["when"]
        if not isinstance(conditions, dict):
            raise ScientificContractError(f"{name}.when must be an object")
        unknown = set(conditions) - set(gate_outcomes)
        if unknown:
            raise ScientificContractError(f"{name}.when names unknown gates: {sorted(unknown)}")
        normalized_when = {}
        for gate_id, outcomes in conditions.items():
            if not isinstance(outcomes, list) or not outcomes:
                raise ScientificContractError(f"{name}.when.{gate_id} must be non-empty")
            normalized = [_token(item, f"{name}.when.{gate_id}[]") for item in outcomes]
            if len(normalized) != len(set(normalized)) \
                    or not set(normalized) <= gate_outcomes[gate_id]:
                raise ScientificContractError(f"{name}.when.{gate_id} is invalid")
            normalized_when[gate_id] = normalized
        result.append({
            "id": identifier,
            "when": normalized_when,
            "terminal": rule["terminal"],
        })
    return result


def _matching_rules(rules: list[dict[str, Any]], outcomes: dict[str, str]) -> list[dict[str, Any]]:
    return [
        rule for rule in rules
        if all(outcomes[gate_id] in allowed for gate_id, allowed in rule["when"].items())
    ]


def _enumerated_decisions(gates: list[dict[str, Any]], rules: list[dict[str, Any]]) \
        -> dict[tuple[str, ...], str]:
    gate_ids = [gate["id"] for gate in gates]
    decisions = {}
    for values in itertools.product(*(gate["outcomes"] for gate in gates)):
        outcome_map = dict(zip(gate_ids, values))
        matches = _matching_rules(rules, outcome_map)
        if len(matches) != 1:
            raise ScientificContractError(
                "decision table must match every outcome combination exactly once; "
                f"outcomes={outcome_map} matches={[item['id'] for item in matches]}"
            )
        decisions[values] = matches[0]["terminal"]
    return decisions


def _typed_policy_decision(
    gates: list[dict[str, Any]], outcomes: dict[str, str],
) -> tuple[str, str]:
    """Apply the fixed validity, decisive, then precision precedence contract."""
    for gate in gates:
        if gate["decision_role"] == "validity_veto" \
                and outcomes[gate["id"]] != gate["neutral_outcome"]:
            return "inconclusive", "policy_validity_veto"

    decisive_labels = []
    for gate in gates:
        if gate["decision_role"] != "decisive":
            continue
        mapping = DECISIVE_OUTCOME_LABELS.get(gate["type"])
        if mapping is None or set(mapping) != set(gate["outcomes"]):
            raise ScientificContractError(
                "typed gate precedence supports decisive materiality or safety gates only"
            )
        decisive_labels.append(mapping[outcomes[gate["id"]]])
    if not decisive_labels:
        raise ScientificContractError(
            "typed gate precedence requires at least one decisive gate"
        )
    if "fail" in decisive_labels:
        return "fail", "policy_decisive_fail"
    if "inconclusive" in decisive_labels:
        return "inconclusive", "policy_decisive_inconclusive"
    if any(
        gate["decision_role"] == "inconclusive_only"
        and outcomes[gate["id"]] != gate["neutral_outcome"]
        for gate in gates
    ):
        return "inconclusive", "policy_precision_inconclusive"
    return "pass", "policy_decisive_pass"


def _policy_decisions(
    gates: list[dict[str, Any]], policy: Any,
) -> dict[tuple[str, ...], str]:
    if policy not in DECISION_POLICIES:
        raise ScientificContractError(
            f"decision_table.policy must be one of {sorted(DECISION_POLICIES)}"
        )
    gate_ids = [gate["id"] for gate in gates]
    decisions = {}
    for values in itertools.product(*(gate["outcomes"] for gate in gates)):
        label, _ = _typed_policy_decision(gates, dict(zip(gate_ids, values)))
        decisions[values] = label
    return decisions


def _validate_role_semantics(
    gates: list[dict[str, Any]], decisions: dict[tuple[str, ...], str],
) -> None:
    for gate_index, gate in enumerate(gates):
        role = gate["decision_role"]
        if role == "decisive":
            continue
        other_indexes = [index for index in range(len(gates)) if index != gate_index]
        other_domains = [gates[index]["outcomes"] for index in other_indexes]
        for other_values in itertools.product(*other_domains):
            def terminal(outcome: str) -> str:
                values = [None] * len(gates)
                values[gate_index] = outcome
                for index, value in zip(other_indexes, other_values):
                    values[index] = value
                return decisions[tuple(values)]

            baseline = terminal(gate["neutral_outcome"])
            observed = {outcome: terminal(outcome) for outcome in gate["outcomes"]}
            if role == "descriptive" and len(set(observed.values())) != 1:
                raise ScientificContractError(
                    f"descriptive gate {gate['id']} changes the terminal decision"
                )
            if role == "validity_veto":
                for outcome, candidate in observed.items():
                    if outcome != gate["neutral_outcome"] \
                            and candidate != "inconclusive":
                        raise ScientificContractError(
                            f"validity_veto gate {gate['id']} does not force "
                            f"inconclusive for outcome {outcome}"
                        )
            if role == "inconclusive_only":
                for outcome, candidate in observed.items():
                    if outcome == gate["neutral_outcome"]:
                        continue
                    allowed = {
                        "pass": {"pass", "inconclusive"},
                        "fail": {"fail"},
                        "inconclusive": {"inconclusive"},
                    }[baseline]
                    if candidate not in allowed:
                        raise ScientificContractError(
                            f"inconclusive_only gate {gate['id']} changes {baseline} "
                            f"to {candidate} for outcome {outcome}"
                        )


def validate_scientific_contract_v2(
    value: Any, route_id: str, operation_id: str,
) -> dict[str, Any]:
    expected = {
        "schema_version", "route_id", "operation_id", "question", "population",
        "intervention", "primary_estimand", "controls", "uncertainty", "gates",
        "competing_explanation", "decision_table", "disabled_actions",
    }
    item = _object(value, expected, "scientific contract")
    if item["schema_version"] != 2:
        raise ScientificContractError("scientific contract schema_version must equal 2")
    if item["route_id"] != route_id or item["operation_id"] != operation_id:
        raise ScientificContractError("scientific contract identity mismatch")
    population = _validate_population(item["population"])
    estimand = _validate_estimand(item["primary_estimand"], population)
    uncertainty = _validate_uncertainty(item["uncertainty"], population)
    gates = _validate_gates(item["gates"], uncertainty)
    allowed_estimands = {estimand["id"]}
    if any(gate["estimand_id"] not in allowed_estimands for gate in gates):
        raise ScientificContractError("every gate must bind the primary estimand id")
    decision_table = item["decision_table"]
    if not isinstance(decision_table, dict) or frozenset(decision_table) not in {
        frozenset({"terminal_actions", "rules"}),
        frozenset({"terminal_actions", "policy"}),
    }:
        raise ScientificContractError(
            "decision_table must contain terminal_actions and exactly one of rules or policy"
        )
    actions = _validate_terminal_actions(
        decision_table["terminal_actions"], population["evidence_role"],
    )
    if "rules" in decision_table:
        decision_encoding = {
            "rules": _normalize_rules(decision_table["rules"], gates),
        }
        decisions = _enumerated_decisions(gates, decision_encoding["rules"])
    else:
        decision_encoding = {"policy": decision_table["policy"]}
        decisions = _policy_decisions(gates, decision_encoding["policy"])
    if set(decisions.values()) != set(TERMINAL_LABELS):
        raise ScientificContractError(
            "decision table must make pass, fail, and inconclusive reachable"
        )
    _validate_role_semantics(gates, decisions)
    intervention = _object(item["intervention"], {
        "primary_variable", "reference", "matched_budget", "fixed_factors",
    }, "intervention")
    fixed = intervention["fixed_factors"]
    if not isinstance(fixed, list) or not fixed \
            or any(not isinstance(value, str) or not value.strip() for value in fixed):
        raise ScientificContractError("intervention.fixed_factors must be non-empty")
    controls = item["controls"]
    if not isinstance(controls, list) or not controls \
            or any(not isinstance(value, str) or not value.strip() for value in controls):
        raise ScientificContractError("controls must be non-empty")
    disabled = item["disabled_actions"]
    if not isinstance(disabled, list) or not disabled \
            or any(not isinstance(value, str) or not value.strip() for value in disabled):
        raise ScientificContractError("disabled_actions must be non-empty")
    return {
        "schema_version": 2,
        "route_id": route_id,
        "operation_id": operation_id,
        "question": _text(item["question"], "question", 16, 2048),
        "population": population,
        "intervention": {
            "primary_variable": _text(
                intervention["primary_variable"], "intervention.primary_variable", 8, 2048,
            ),
            "reference": _text(
                intervention["reference"], "intervention.reference", 8, 2048,
            ),
            "matched_budget": _text(
                intervention["matched_budget"], "intervention.matched_budget", 8, 2048,
            ),
            "fixed_factors": [value.strip() for value in fixed],
        },
        "primary_estimand": estimand,
        "controls": [value.strip() for value in controls],
        "uncertainty": uncertainty,
        "gates": gates,
        "competing_explanation": _text(
            item["competing_explanation"], "competing_explanation", 16, 2048,
        ),
        "decision_table": {
            "terminal_actions": actions,
            **decision_encoding,
        },
        "disabled_actions": [value.strip() for value in disabled],
    }


def validate_scientific_contract_v3(
    value: Any, route_id: str, operation_id: str, *,
    expected_snapshot_commit: str | None = None,
    read_evidence_file: Callable[[str], bytes] | None = None,
    require_current_design: bool = False,
) -> dict[str, Any]:
    expected = {
        "schema_version", "route_id", "operation_id", "question", "population",
        "intervention", "primary_estimand", "controls", "uncertainty", "gates",
        "competing_explanation", "decision_table", "disabled_actions",
        "research_update_binding",
    }
    item = _object(value, expected, "scientific contract")
    if item["schema_version"] != 3:
        raise ScientificContractError("scientific contract schema_version must equal 3")
    legacy_value = {
        key: field for key, field in item.items()
        if key != "research_update_binding"
    }
    legacy_value["schema_version"] = 2
    validated = validate_scientific_contract_v2(
        legacy_value, route_id, operation_id,
    )
    return {
        **validated,
        "schema_version": 3,
        "research_update_binding": _validate_research_update_binding(
            item["research_update_binding"],
            expected_snapshot_commit=expected_snapshot_commit,
            read_evidence_file=read_evidence_file,
            require_current_design=require_current_design,
        ),
    }


def scientific_terminal_tuples(contract: dict[str, Any]) -> list[dict[str, str]]:
    if contract.get("schema_version") not in {2, 3}:
        raise ScientificContractError("terminal derivation requires scientific schema 2 or 3")
    return [
        dict(contract["decision_table"]["terminal_actions"][label]["terminal"])
        for label in TERMINAL_LABELS
    ]


def evaluate_gate_outcomes(
    contract: dict[str, Any], gate_outcomes: Any,
) -> dict[str, Any]:
    if contract.get("schema_version") not in {2, 3}:
        raise ScientificContractError(
            "generic gate evaluation requires scientific schema 2 or 3"
        )
    gates = contract["gates"]
    if not isinstance(gate_outcomes, dict) \
            or set(gate_outcomes) != {gate["id"] for gate in gates}:
        raise ScientificContractError("gate outcome ids must exactly match the contract gates")
    normalized = {}
    for gate in gates:
        outcome = gate_outcomes[gate["id"]]
        if outcome not in gate["outcomes"]:
            raise ScientificContractError(
                f"gate outcome is invalid: {gate['id']}={outcome}"
            )
        normalized[gate["id"]] = outcome
    decision_table = contract["decision_table"]
    if "rules" in decision_table:
        matches = _matching_rules(decision_table["rules"], normalized)
        if len(matches) != 1:
            raise ScientificContractError("validated decision table did not resolve uniquely")
        label = matches[0]["terminal"]
        decision_rule_id = matches[0]["id"]
    else:
        label, decision_rule_id = _typed_policy_decision(gates, normalized)
    action = contract["decision_table"]["terminal_actions"][label]
    return {
        **action["terminal"],
        "terminal_label": label,
        "decision_rule_id": decision_rule_id,
        "next_action_id": action["next_action_id"],
        "family_effect": action["family_effect"],
        "gate_outcomes": normalized,
    }
