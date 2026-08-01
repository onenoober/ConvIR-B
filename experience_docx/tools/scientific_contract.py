#!/usr/bin/env python3
"""Pure validation and evaluation for forward scientific contracts."""

from __future__ import annotations

import itertools
import json
import re
from typing import Any


class ScientificContractError(ValueError):
    pass


SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
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
    decision_table = _object(
        item["decision_table"], {"terminal_actions", "rules"}, "decision_table",
    )
    actions = _validate_terminal_actions(
        decision_table["terminal_actions"], population["evidence_role"],
    )
    rules = _normalize_rules(decision_table["rules"], gates)
    decisions = _enumerated_decisions(gates, rules)
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
            "rules": rules,
        },
        "disabled_actions": [value.strip() for value in disabled],
    }


def scientific_terminal_tuples(contract: dict[str, Any]) -> list[dict[str, str]]:
    if contract.get("schema_version") != 2:
        raise ScientificContractError("terminal derivation requires scientific schema 2")
    return [
        dict(contract["decision_table"]["terminal_actions"][label]["terminal"])
        for label in TERMINAL_LABELS
    ]


def evaluate_gate_outcomes(
    contract: dict[str, Any], gate_outcomes: Any,
) -> dict[str, Any]:
    if contract.get("schema_version") != 2:
        raise ScientificContractError("generic gate evaluation requires scientific schema 2")
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
    matches = _matching_rules(contract["decision_table"]["rules"], normalized)
    if len(matches) != 1:
        raise ScientificContractError("validated decision table did not resolve uniquely")
    label = matches[0]["terminal"]
    action = contract["decision_table"]["terminal_actions"][label]
    return {
        **action["terminal"],
        "terminal_label": label,
        "decision_rule_id": matches[0]["id"],
        "next_action_id": action["next_action_id"],
        "family_effect": action["family_effect"],
        "gate_outcomes": normalized,
    }
