"""Tests for the pure scientific decision contract."""

import hashlib
import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path


TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import scientific_contract as SCIENCE  # noqa: E402
import route_runtime_contract as RUNTIME  # noqa: E402


def contract():
    return {
        "schema_version": 2,
        "route_id": "route",
        "operation_id": "S0",
        "question": "Does the frozen effect satisfy the materiality threshold?",
        "population": {
            "evidence_role": "development_screening",
            "grouping_unit": "scene",
            "independent_group_count": 200,
            "strata": [
                {"id": "indoor", "independent_group_count": 100},
                {"id": "outdoor", "independent_group_count": 100},
            ],
            "allow_confirmation": False,
            "allow_canary": False,
            "allow_locked_test": False,
        },
        "intervention": {
            "primary_variable": "frozen intervention",
            "reference": "frozen baseline",
            "matched_budget": "one matched observation per scene",
            "fixed_factors": ["model and preprocessing are frozen"],
        },
        "primary_estimand": {
            "id": "scene_mean_error",
            "metric_id": "psnr_error_db",
            "direction": "lower",
            "aggregation": "separate stratum means followed by the frozen maximum",
            "unit": "scene",
            "strata": ["indoor", "outdoor"],
        },
        "controls": ["paired baseline identity"],
        "uncertainty": {
            "id": "simultaneous_ci",
            "method_id": "group_bootstrap",
            "confidence_level": 0.95,
            "independent_unit": "scene",
            "comparison_family": "indoor_outdoor",
        },
        "gates": [
            {
                "id": "materiality",
                "type": "materiality",
                "estimand_id": "scene_mean_error",
                "reference": "simultaneous interval against 0.05 dB",
                "direction": "max",
                "threshold": 0.05,
                "uncertainty_id": "simultaneous_ci",
                "comparison_family": "indoor_outdoor",
                "decision_role": "decisive",
                "outcomes": [
                    "favorable", "unfavorable", "indeterminate", "invalid",
                ],
                "neutral_outcome": "favorable",
            },
            {
                "id": "precision",
                "type": "precision",
                "estimand_id": "scene_mean_error",
                "reference": "maximum simultaneous interval distance",
                "direction": "max",
                "threshold": 0.025,
                "uncertainty_id": "simultaneous_ci",
                "comparison_family": "indoor_outdoor",
                "decision_role": "inconclusive_only",
                "outcomes": ["met", "unmet", "invalid"],
                "neutral_outcome": "met",
            },
        ],
        "competing_explanation": "Observed variation may differ between the frozen strata.",
        "decision_table": {
            "terminal_actions": {
                "pass": {
                    "terminal": {
                        "state": "COMPLETED_GATE_PASS",
                        "decision": "MEASUREMENT_PASS",
                        "authorizes": "RESULT_ONLY",
                    },
                    "next_action_id": "report_result",
                    "family_effect": "advance",
                },
                "fail": {
                    "terminal": {
                        "state": "COMPLETED_GATE_FAIL",
                        "decision": "MEASUREMENT_FAIL",
                        "authorizes": "NONE",
                    },
                    "next_action_id": "stop_claim",
                    "family_effect": "stop",
                },
                "inconclusive": {
                    "terminal": {
                        "state": "COMPLETED_INCONCLUSIVE",
                        "decision": "MEASUREMENT_INCONCLUSIVE",
                        "authorizes": "NONE",
                    },
                    "next_action_id": None,
                    "family_effect": "record_only",
                },
            },
            "rules": [
                {
                    "id": "bad_side_fails",
                    "when": {"materiality": ["unfavorable"]},
                    "terminal": "fail",
                },
                {
                    "id": "good_and_precise_passes",
                    "when": {
                        "materiality": ["favorable"],
                        "precision": ["met"],
                    },
                    "terminal": "pass",
                },
                {
                    "id": "good_but_imprecise_is_inconclusive",
                    "when": {
                        "materiality": ["favorable"],
                        "precision": ["unmet", "invalid"],
                    },
                    "terminal": "inconclusive",
                },
                {
                    "id": "threshold_unresolved",
                    "when": {"materiality": ["indeterminate", "invalid"]},
                    "terminal": "inconclusive",
                },
            ],
        },
        "disabled_actions": ["no result-adaptive extension"],
    }


def contract_v3(snapshot_commit="a" * 40, terminal_record_sha256="b" * 64):
    value = contract()
    value["schema_version"] = 3
    value["research_update_binding"] = {
        "snapshot_commit": snapshot_commit,
        "trigger_type": "post_terminal",
        "trigger_terminals": [{
            "route_id": "prior-route",
            "terminal_record_sha256": terminal_record_sha256,
        }],
        "bottleneck_class": "scientific_hypothesis",
        "bottleneck_statement": (
            "The archived effect is valid but too small to support the current mechanism."
        ),
        "literature_basis": [{
            "identifier": "doi:10.0000/example",
            "source_status": "peer_reviewed",
            "task": "paired image restoration",
            "transferable_claim": (
                "Conditional processing can separate scene-dependent restoration effects."
            ),
            "applicability_limit": (
                "The published protocol does not establish an effect on this project dataset."
            ),
        }],
        "hypotheses": [
            {
                "id": "global_capacity",
                "statement": "The global restoration path lacks sufficient conditional capacity.",
                "discriminating_prediction": (
                    "A conditional multi-arm intervention improves the frozen estimand."
                ),
                "falsifier": (
                    "Every conditional arm remains below the frozen worthwhile-effect margin."
                ),
            },
            {
                "id": "measurement_mismatch",
                "statement": "The current aggregate hides a stratum-specific usable effect.",
                "discriminating_prediction": (
                    "Predeclared strata show opposing effects under the same intervention."
                ),
                "falsifier": (
                    "All simultaneous stratum intervals remain practically equivalent."
                ),
            },
        ],
        "design_selection": {
            "strategy": "multi_arm",
            "decision_value": "Distinguishes both live hypotheses in one frozen comparison family.",
            "expected_time_to_decision": "One shared inference pass plus grouped uncertainty.",
            "shared_setup": "All arms reuse the same paired scenes and frozen preprocessing.",
            "worst_case_stopping_cost": "Stop after the predeclared complete arm set.",
        },
    }
    return value


def terminal_index_bytes():
    raw_line = json.dumps(
        {"route_id": "prior-route"}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return raw_line + b"\n", hashlib.sha256(raw_line).hexdigest()


def precision_certificate():
    return {
        "schema_version": 2,
        "certificate_id": "route_precision",
        "route_id": "route",
        "operation_id": "S0",
        "primary_estimand_id": "scene_mean_error",
        "independent_unit": "scene",
        "comparison_family": "indoor_outdoor",
        "method": "normal_mean",
        "confidence_level": 0.95,
        "critical_value": 2.241402727604947,
        "target_half_width": 0.025,
        "assurance": {
            "method_id": "external_sd_upper_bound",
            "probability": 0.8,
            "planning_sd_rule": "Use the frozen pre-run upper confidence bound for each stratum SD.",
        },
        "strata": [
            {
                "id": identifier,
                "independent_groups_available": 100,
                "independent_groups_planned": 100,
                "planning_sd": 0.09,
                "planning_sd_upper_bound": 0.1,
                "independent_groups_required": 81,
                "feasible": True,
                "source_reference": f"Frozen external planning evidence for {identifier}.",
            }
            for identifier in ("indoor", "outdoor")
        ],
        "feasible": True,
        "source_role": "development_screening",
        "source_reference": "Frozen external planning evidence independent of route outcomes.",
    }


class ScientificContractTests(unittest.TestCase):
    def test_schema3_binds_terminal_hypotheses_and_preserves_gate_semantics(self):
        raw_index, terminal_sha = terminal_index_bytes()
        value = SCIENCE.validate_scientific_contract_v3(
            contract_v3(terminal_record_sha256=terminal_sha), "route", "S0",
            expected_snapshot_commit="a" * 40,
            read_evidence_file=lambda _: raw_index,
        )
        result = SCIENCE.evaluate_gate_outcomes(value, {
            "materiality": "unfavorable", "precision": "unmet",
        })
        self.assertEqual("fail", result["terminal_label"])
        self.assertEqual("multi_arm", value["research_update_binding"][
            "design_selection"
        ]["strategy"])
        spec = {
            "route_id": "route", "operation_id": "S0",
            "precision_contract": {"mode": "formal_precision"},
        }
        self.assertTrue(RUNTIME.validate_precision_certificate(
            precision_certificate(), spec, value,
        )["feasible"])

    def test_schema3_rejects_missing_or_nonfalsifiable_research_binding(self):
        value = contract_v3()
        del value["research_update_binding"]
        with self.assertRaisesRegex(SCIENCE.ScientificContractError, "contain exactly"):
            SCIENCE.validate_scientific_contract_v3(value, "route", "S0")

        value = contract_v3()
        value["research_update_binding"]["hypotheses"][1]["id"] = "global_capacity"
        with self.assertRaisesRegex(SCIENCE.ScientificContractError, "ids must be unique"):
            SCIENCE.validate_scientific_contract_v3(value, "route", "S0")

        value = contract_v3()
        value["research_update_binding"]["hypotheses"][0]["falsifier"] = "none"
        with self.assertRaisesRegex(SCIENCE.ScientificContractError, "16-2048"):
            SCIENCE.validate_scientific_contract_v3(value, "route", "S0")

        value = contract_v3()
        value["research_update_binding"]["literature_basis"] *= 2
        with self.assertRaisesRegex(
            SCIENCE.ScientificContractError, "literature identifiers must be unique",
        ):
            SCIENCE.validate_scientific_contract_v3(value, "route", "S0")

    def test_schema3_rejects_wrong_snapshot_or_terminal_record_sha(self):
        raw_index, terminal_sha = terminal_index_bytes()
        with self.assertRaisesRegex(SCIENCE.ScientificContractError, "authoritative main"):
            SCIENCE.validate_scientific_contract_v3(
                contract_v3("c" * 40, terminal_sha), "route", "S0",
                expected_snapshot_commit="a" * 40,
                read_evidence_file=lambda _: raw_index,
            )
        with self.assertRaisesRegex(SCIENCE.ScientificContractError, "matching authoritative"):
            SCIENCE.validate_scientific_contract_v3(
                contract_v3(terminal_record_sha256="d" * 64), "route", "S0",
                expected_snapshot_commit="a" * 40,
                read_evidence_file=lambda _: raw_index,
            )

    def test_schema3_program_foundation_has_no_terminal_dependency(self):
        value = contract_v3()
        binding = value["research_update_binding"]
        binding["trigger_type"] = "program_foundation"
        binding["trigger_terminals"] = []
        validated = SCIENCE.validate_scientific_contract_v3(
            value, "route", "S0", expected_snapshot_commit="a" * 40,
            read_evidence_file=lambda _: (_ for _ in ()).throw(
                AssertionError("foundation must not read a terminal index")
            ),
        )
        self.assertEqual(
            "program_foundation",
            validated["research_update_binding"]["trigger_type"],
        )

        value = contract_v3()
        value["research_update_binding"]["trigger_terminals"] = []
        with self.assertRaisesRegex(
            SCIENCE.ScientificContractError, "post_terminal or zero entries",
        ):
            SCIENCE.validate_scientific_contract_v3(value, "route", "S0")

        value = contract_v3()
        value["research_update_binding"]["trigger_type"] = "program_foundation"
        with self.assertRaisesRegex(
            SCIENCE.ScientificContractError, "post_terminal or zero entries",
        ):
            SCIENCE.validate_scientific_contract_v3(value, "route", "S0")
    def test_complete_table_is_valid_and_terminal_tuples_are_derived(self):
        value = SCIENCE.validate_scientific_contract_v2(contract(), "route", "S0")
        terminals = SCIENCE.scientific_terminal_tuples(value)
        self.assertEqual(3, len(terminals))
        self.assertEqual("MEASUREMENT_FAIL", terminals[1]["decision"])

    def test_generic_evaluator_preserves_decisive_bad_side(self):
        value = SCIENCE.validate_scientific_contract_v2(contract(), "route", "S0")
        result = SCIENCE.evaluate_gate_outcomes(value, {
            "materiality": "unfavorable",
            "precision": "unmet",
        })
        self.assertEqual("COMPLETED_GATE_FAIL", result["state"])
        self.assertEqual("bad_side_fails", result["decision_rule_id"])

    def test_precision_first_table_is_rejected(self):
        value = contract()
        value["decision_table"]["rules"] = [
            {
                "id": "precision_veto",
                "when": {"precision": ["unmet", "invalid"]},
                "terminal": "inconclusive",
            },
            {
                "id": "bad_precise",
                "when": {
                    "materiality": ["unfavorable"], "precision": ["met"],
                },
                "terminal": "fail",
            },
            {
                "id": "good_precise",
                "when": {
                    "materiality": ["favorable"], "precision": ["met"],
                },
                "terminal": "pass",
            },
            {
                "id": "unresolved_precise",
                "when": {
                    "materiality": ["indeterminate", "invalid"],
                    "precision": ["met"],
                },
                "terminal": "inconclusive",
            },
        ]
        with self.assertRaisesRegex(
            SCIENCE.ScientificContractError,
            "inconclusive_only gate precision changes fail to inconclusive",
        ):
            SCIENCE.validate_scientific_contract_v2(value, "route", "S0")

    def test_precision_gate_cannot_be_a_decisive_fail_source(self):
        value = contract()
        value["gates"][1]["decision_role"] = "decisive"
        with self.assertRaisesRegex(
            SCIENCE.ScientificContractError, "precision cannot be decisive",
        ):
            SCIENCE.validate_scientific_contract_v2(value, "route", "S0")

    def test_validity_and_precision_neutral_outcomes_cannot_be_reversed(self):
        value = contract()
        value["gates"][1]["neutral_outcome"] = "unmet"
        with self.assertRaisesRegex(
            SCIENCE.ScientificContractError, "neutral_outcome must equal met",
        ):
            SCIENCE.validate_scientific_contract_v2(value, "route", "S0")

        value = contract()
        value["gates"].append({
            "id": "coverage",
            "type": "coverage",
            "estimand_id": "scene_mean_error",
            "reference": "frozen complete independent-unit coverage",
            "direction": "equal",
            "threshold": True,
            "uncertainty_id": "simultaneous_ci",
            "comparison_family": "indoor_outdoor",
            "decision_role": "validity_veto",
            "outcomes": ["pass", "fail", "invalid"],
            "neutral_outcome": "fail",
        })
        with self.assertRaisesRegex(
            SCIENCE.ScientificContractError, "neutral_outcome must equal pass",
        ):
            SCIENCE.validate_scientific_contract_v2(value, "route", "S0")

    def test_validity_veto_forces_inconclusive_over_scientific_fail(self):
        value = contract()
        value["gates"].append({
            "id": "integrity",
            "type": "integrity",
            "estimand_id": "scene_mean_error",
            "reference": "frozen identity and complete-unit checks",
            "direction": "equal",
            "threshold": True,
            "uncertainty_id": "simultaneous_ci",
            "comparison_family": "indoor_outdoor",
            "decision_role": "validity_veto",
            "outcomes": ["pass", "fail", "invalid"],
            "neutral_outcome": "pass",
        })
        value["decision_table"]["rules"].insert(0, {
            "id": "invalid_science",
            "when": {"integrity": ["fail", "invalid"]},
            "terminal": "inconclusive",
        })
        for rule in value["decision_table"]["rules"][1:]:
            rule["when"]["integrity"] = ["pass"]
        validated = SCIENCE.validate_scientific_contract_v2(value, "route", "S0")
        result = SCIENCE.evaluate_gate_outcomes(validated, {
            "materiality": "unfavorable", "precision": "met",
            "integrity": "fail",
        })
        self.assertEqual("inconclusive", result["terminal_label"])
        value["decision_table"]["rules"][0]["terminal"] = "fail"
        with self.assertRaisesRegex(
            SCIENCE.ScientificContractError, "validity_veto gate integrity",
        ):
            SCIENCE.validate_scientific_contract_v2(value, "route", "S0")

    def test_overlapping_or_incomplete_table_is_rejected(self):
        value = contract()
        value["decision_table"]["rules"].append({
            "id": "overlapping_default", "when": {}, "terminal": "inconclusive",
        })
        with self.assertRaisesRegex(
            SCIENCE.ScientificContractError, "exactly once",
        ):
            SCIENCE.validate_scientific_contract_v2(value, "route", "S0")

    def test_stratum_counts_are_identity_bound(self):
        value = contract()
        value["population"]["independent_group_count"] = 201
        with self.assertRaisesRegex(
            SCIENCE.ScientificContractError, "stratum counts must sum",
        ):
            SCIENCE.validate_scientific_contract_v2(value, "route", "S0")

    def test_gate_result_ids_and_outcomes_are_exact(self):
        value = SCIENCE.validate_scientific_contract_v2(contract(), "route", "S0")
        with self.assertRaisesRegex(
            SCIENCE.ScientificContractError, "exactly match",
        ):
            SCIENCE.evaluate_gate_outcomes(value, {"materiality": "favorable"})
        with self.assertRaisesRegex(
            SCIENCE.ScientificContractError, "gate outcome is invalid",
        ):
            SCIENCE.evaluate_gate_outcomes(value, {
                "materiality": "favorable", "precision": "precise",
            })

    def test_descriptive_gate_cannot_change_terminal(self):
        value = contract()
        descriptive = deepcopy(value["gates"][1])
        descriptive["decision_role"] = "descriptive"
        value["gates"][1] = descriptive
        with self.assertRaisesRegex(
            SCIENCE.ScientificContractError, "descriptive gate precision",
        ):
            SCIENCE.validate_scientific_contract_v2(value, "route", "S0")

    def test_precision_schema2_is_bound_to_estimand_and_each_stratum(self):
        scientific = SCIENCE.validate_scientific_contract_v2(contract(), "route", "S0")
        spec = {
            "route_id": "route", "operation_id": "S0",
            "precision_contract": {"mode": "formal_precision"},
        }
        result = RUNTIME.validate_precision_certificate(
            precision_certificate(), spec, scientific,
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(["indoor", "outdoor"], [item["id"] for item in result["strata"]])

    def test_precision_schema2_rejects_route_wide_or_point_sd_substitution(self):
        scientific = SCIENCE.validate_scientific_contract_v2(contract(), "route", "S0")
        spec = {
            "route_id": "route", "operation_id": "S0",
            "precision_contract": {"mode": "formal_precision"},
        }
        value = precision_certificate()
        value["strata"][0]["independent_groups_available"] = 200
        with self.assertRaisesRegex(RUNTIME.ContractError, "differ from the population"):
            RUNTIME.validate_precision_certificate(value, spec, scientific)
        value = precision_certificate()
        value["strata"][0]["independent_groups_required"] = 50
        with self.assertRaisesRegex(RUNTIME.ContractError, "upper-bound calculation"):
            RUNTIME.validate_precision_certificate(value, spec, scientific)

        value = precision_certificate()
        value["critical_value"] = 1.96
        with self.assertRaisesRegex(RUNTIME.ContractError, "simultaneous Bonferroni"):
            RUNTIME.validate_precision_certificate(value, spec, scientific)

    def test_terminal_labels_must_encode_real_action_value(self):
        value = contract()
        for action in value["decision_table"]["terminal_actions"].values():
            action["next_action_id"] = None
            action["family_effect"] = "record_only"
            action["terminal"]["authorizes"] = "NONE"
        with self.assertRaisesRegex(
            SCIENCE.ScientificContractError, "each terminal outcome must change",
        ):
            SCIENCE.validate_scientific_contract_v2(value, "route", "S0")

    def test_two_labels_cannot_share_the_same_effective_action(self):
        value = contract()
        inconclusive = value["decision_table"]["terminal_actions"]["inconclusive"]
        inconclusive["next_action_id"] = "stop_claim"
        inconclusive["family_effect"] = "stop"
        with self.assertRaisesRegex(
            SCIENCE.ScientificContractError, "each terminal outcome must change",
        ):
            SCIENCE.validate_scientific_contract_v2(value, "route", "S0")


if __name__ == "__main__":
    unittest.main()
