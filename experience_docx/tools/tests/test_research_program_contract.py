"""Contract tests for flexible program-level route governance."""

import copy
import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import research_program_contract as PROGRAM  # noqa: E402


def contract(*, state="open", budget=3, used=1, amendments=None):
    return {
        "schema_version": 1,
        "program_id": "restoration_program",
        "objective": "Determine which scientifically distinct route can improve regional restoration.",
        "source_anchors": [{
            "id": "official_anchor",
            "reference": "immutable upstream architecture commit and checkpoint identity",
            "allowed_uses": ["baseline", "initialization", "comparison", "diagnostic"],
        }],
        "dependency_roles": {
            "official_baseline": ["baseline", "initialization", "comparison"],
            "diagnostic_only": ["diagnostic", "comparison"],
        },
        "stages": {
            "development": {
                "evidence_roles": ["engineering_debug", "development_screening"],
                "protected_permissions": {
                    "allow_confirmation": False,
                    "allow_canary": False,
                    "allow_locked_test": False,
                },
                "route_families": ["selector_family"],
            },
        },
        "route_families": {
            "selector_family": {
                "core_assumption": "observable regional features identify the useful action",
                "adjacent_budget": budget,
                "attempts_used": used,
                "state": state,
                "close_condition": "close when the preregistered specificity and safety gates fail",
                "reopen_evidence_types": ["new_observable_information", "target_realignment"],
            },
        },
        "orthogonal_dimensions": [
            "measurement_target", "region_definition", "information_source",
            "decision_semantics",
        ],
        "amendments": amendments or [],
    }


def claim(*, mechanism="adjacent", sequence=2, role="official_baseline",
          use="baseline", permissions=None, orthogonal=None, reopen=None):
    return {
        "program_id": "restoration_program",
        "stage_id": "development",
        "family_id": "selector_family",
        "mechanism_type": mechanism,
        "evidence_role": "development_screening",
        "protected_permissions": permissions or {
            "allow_confirmation": False, "allow_canary": False,
            "allow_locked_test": False,
        },
        "dependencies": [{
            "role": role, "use": use, "anchor_id": "official_anchor",
        }],
        "adjacent_sequence": sequence,
        "orthogonal_changes": orthogonal or [],
        "reopen_evidence": reopen or [],
    }


class ResearchProgramContractTests(unittest.TestCase):
    def test_adjacent_route_consumes_program_defined_budget(self):
        program = PROGRAM.validate_program_contract(contract(budget=7, used=5))
        route = claim(sequence=6)
        result = PROGRAM.validate_route_authorization(program, route)
        self.assertTrue(result["adjacent_budget_consumed"])

    def test_adjacent_budget_is_not_a_global_two_attempt_rule(self):
        program = PROGRAM.validate_program_contract(contract(budget=9, used=7))
        result = PROGRAM.validate_route_authorization(program, claim(sequence=8))
        self.assertEqual("adjacent", result["mechanism_type"])

    def test_exhausted_or_closed_family_rejects_adjacent_route(self):
        exhausted = PROGRAM.validate_program_contract(contract(budget=2, used=2))
        with self.assertRaises(PROGRAM.ProgramContractError):
            PROGRAM.validate_route_authorization(exhausted, claim(sequence=3))
        closed = PROGRAM.validate_program_contract(contract(state="closed"))
        with self.assertRaises(PROGRAM.ProgramContractError):
            PROGRAM.validate_route_authorization(closed, claim())

    def test_orthogonal_route_can_escape_a_closed_adjacent_family(self):
        program = PROGRAM.validate_program_contract(contract(state="closed"))
        route = claim(
            mechanism="orthogonal", sequence=None,
            orthogonal=[{
                "dimension": "measurement_target",
                "reason": "changes the measured scientific target rather than retuning the selector",
            }],
        )
        result = PROGRAM.validate_route_authorization(program, route)
        self.assertFalse(result["adjacent_budget_consumed"])

    def test_orthogonal_route_requires_an_allowed_substantive_dimension(self):
        program = PROGRAM.validate_program_contract(contract())
        route = claim(
            mechanism="orthogonal", sequence=None,
            orthogonal=[{"dimension": "learning_rate", "reason": "a parameter-only change"}],
        )
        with self.assertRaises(PROGRAM.ProgramContractError):
            PROGRAM.validate_route_authorization(program, route)

    def test_closed_family_reopen_requires_allowed_present_evidence(self):
        program = PROGRAM.validate_program_contract(contract(state="closed"))
        route = claim(
            mechanism="reopen", sequence=None,
            reopen=[{
                "type": "new_observable_information",
                "relpath": "experience_docx/experiment_logs/new_evidence/closeout.json",
            }],
        )
        with self.assertRaises(PROGRAM.ProgramContractError):
            PROGRAM.validate_route_authorization(program, route, evidence_exists=lambda _: False)
        result = PROGRAM.validate_route_authorization(
            program, route, evidence_exists=lambda path: path.endswith("closeout.json"),
        )
        self.assertEqual("reopen", result["mechanism_type"])

    def test_reopen_rejects_an_unapproved_evidence_type(self):
        program = PROGRAM.validate_program_contract(contract(state="closed"))
        route = claim(
            mechanism="reopen", sequence=None,
            reopen=[{
                "type": "more_epochs",
                "relpath": "experience_docx/experiment_logs/new_evidence/closeout.json",
            }],
        )
        with self.assertRaises(PROGRAM.ProgramContractError):
            PROGRAM.validate_route_authorization(program, route, evidence_exists=lambda _: True)

    def test_dependency_role_prevents_silent_scientific_repurposing(self):
        program = PROGRAM.validate_program_contract(contract())
        route = claim(role="diagnostic_only", use="supervision")
        with self.assertRaises(PROGRAM.ProgramContractError):
            PROGRAM.validate_route_authorization(program, route)

    def test_evidence_bound_amendment_can_expand_dependency_use(self):
        amendment = {
            "id": "approve_supervision",
            "reason": "an independent audit established that this source is valid supervision",
            "evidence_refs": ["experience_docx/experiment_logs/audit/closeout.json"],
            "approved_scope": "diagnostic_only role may provide supervision for this program",
            "changes": [{
                "kind": "dependency_uses", "target": "diagnostic_only",
                "value": ["diagnostic", "comparison", "supervision"],
            }],
        }
        value = contract(amendments=[amendment])
        value["source_anchors"][0]["allowed_uses"].append("supervision")
        program = PROGRAM.validate_program_contract(value, evidence_exists=lambda _: True)
        result = PROGRAM.validate_route_authorization(
            program, claim(role="diagnostic_only", use="supervision"),
        )
        self.assertEqual("supervision", result["dependencies"][0]["use"])

    def test_amendment_fails_when_its_evidence_is_missing(self):
        amendment = {
            "id": "more_budget",
            "reason": "a mechanism audit materially changes the expected information gain",
            "evidence_refs": ["experience_docx/experiment_logs/audit/closeout.json"],
            "approved_scope": "one additional adjacent mechanism test",
            "changes": [{"kind": "adjacent_budget", "target": "selector_family", "value": 4}],
        }
        with self.assertRaises(PROGRAM.ProgramContractError):
            PROGRAM.validate_program_contract(
                contract(amendments=[amendment]), evidence_exists=lambda _: False,
            )
        with self.assertRaises(PROGRAM.ProgramContractError):
            PROGRAM.validate_program_contract(contract(amendments=[amendment]))

    def test_stage_permissions_fail_closed_without_blocking_unprotected_work(self):
        program = PROGRAM.validate_program_contract(contract())
        protected = claim(permissions={
            "allow_confirmation": True, "allow_canary": False,
            "allow_locked_test": False,
        })
        with self.assertRaises(PROGRAM.ProgramContractError):
            PROGRAM.validate_route_authorization(program, protected)
        self.assertEqual(
            "development_screening",
            PROGRAM.validate_route_authorization(program, claim())["evidence_role"],
        )

    def test_contract_and_authorization_digests_are_deterministic(self):
        first = PROGRAM.validate_program_contract(contract())
        second = PROGRAM.validate_program_contract(copy.deepcopy(contract()))
        self.assertEqual(first["contract_sha256"], second["contract_sha256"])
        route = claim()
        one = PROGRAM.validate_route_authorization(first, route)
        two = PROGRAM.validate_route_authorization(second, copy.deepcopy(route))
        self.assertEqual(one["authorization_sha256"], two["authorization_sha256"])


if __name__ == "__main__":
    unittest.main()
