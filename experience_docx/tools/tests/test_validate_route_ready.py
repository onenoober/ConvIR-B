"""Tests for route-ready entrypoint and staged-snapshot guards."""

import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import validate_route_ready as READY  # noqa: E402


GOOD = b'''\
from route_program_api import load_context, write_contract_result, write_run_result
def contract(context_path):
    context = load_context(context_path, "contract")
    write_contract_result(context, checks={"ok": True})
def run(context_path):
    context = load_context(context_path, "run")
    write_run_result(context, state="PASS", decision="PASS", authorizes="NEXT")
def main():
    option = "--context"
    if option:
        contract(None)
    else:
        run(None)
'''

SCHEMA2_ENGINEERING = b'''\
from route_program_api import load_context, write_contract_result, write_run_result
def contract(context_path):
    context = load_context(context_path, "contract")
    engineering = {
        "mode": "gpu_synthetic_no_data", "device": "cuda",
        "fixture": {"batch": 1, "channels": 3, "height": 8, "width": 8},
        "production_path_exercised": True, "protected_data_touched": False,
        "scientific_output_created": False, "scientific_training_occurred": False,
    }
    write_contract_result(context, checks={"ok": True}, engineering=engineering)
def run(context_path):
    context = load_context(context_path, "run")
    write_run_result(context, state="PASS", decision="PASS", authorizes="NEXT")
def main():
    option = "--context"
    if option:
        contract(None)
    else:
        run(None)
'''

SCHEMA2_ENGINEERING_COST = SCHEMA2_ENGINEERING.replace(
    b'"scientific_output_created": False, "scientific_training_occurred": False,',
    b'"scientific_output_created": False, "scientific_training_occurred": False,\n'
    b'        "cost": {"observed_iterations": 10, "observed_wall_seconds": 1.0,\n'
    b'                 "observed_peak_memory_mib": 128.0},',
)

SCIENTIFIC_SCHEMA2 = SCHEMA2_ENGINEERING.replace(
    b"write_run_result",
    b"write_gate_result",
).replace(
    b'write_gate_result(context, state="PASS", decision="PASS", authorizes="NEXT")',
    b'write_gate_result(context, gate_outcomes={"materiality": "favorable"})',
)


class RouteReadyTests(unittest.TestCase):
    def test_standard_entrypoint_interface_passes(self):
        READY.check_entrypoint(GOOD, "experience_docx/tools/program.py")

    def test_schema2_complete_engineering_dict_passes(self):
        READY.check_entrypoint(
            SCHEMA2_ENGINEERING, "experience_docx/tools/program.py",
            require_engineering=True,
        )

    def test_schema2_complete_cost_engineering_dict_passes(self):
        READY.check_entrypoint(
            SCHEMA2_ENGINEERING_COST, "experience_docx/tools/program.py",
            require_engineering=True, require_cost_evidence=True,
        )

    def test_schema2_cost_contract_requires_cost_evidence(self):
        with self.assertRaisesRegex(READY.ReadyError, "missing=\\['cost'\\]"):
            READY.check_entrypoint(
                SCHEMA2_ENGINEERING, "experience_docx/tools/program.py",
                require_engineering=True, require_cost_evidence=True,
            )

    def test_schema2_cost_evidence_requires_cost_contract(self):
        with self.assertRaisesRegex(READY.ReadyError, "unknown=\\['cost'\\]"):
            READY.check_entrypoint(
                SCHEMA2_ENGINEERING_COST, "experience_docx/tools/program.py",
                require_engineering=True,
            )

    def test_repo_asset_and_entrypoint_errors_are_reported_together(self):
        asset = {
            "assets": [{
                "id": "model_source", "kind": "file",
                "path": "{REMOTE_REPO}/models/model.py", "sha256": "a" * 64,
            }],
        }
        errors = READY.independent_operation_errors(
            asset=asset,
            read_repo_file=lambda _: b"current model source",
            entrypoint_raw=SCHEMA2_ENGINEERING,
            entrypoint_relpath="experience_docx/tools/program.py",
            require_engineering=True,
            scientific_schema=1,
            require_unit_ledger=False,
            require_cost_evidence=True,
        )
        self.assertEqual(2, len(errors))
        self.assertTrue(any("SHA-256 mismatch" in error for error in errors))
        self.assertTrue(any("missing=['cost']" in error for error in errors))

    def test_scientific_schema2_requires_gate_writer(self):
        READY.check_entrypoint(
            SCIENTIFIC_SCHEMA2, "experience_docx/tools/program.py",
            require_engineering=True, scientific_schema=2,
        )
        with self.assertRaisesRegex(READY.ReadyError, "write_gate_result"):
            READY.check_entrypoint(
                SCHEMA2_ENGINEERING, "experience_docx/tools/program.py",
                require_engineering=True, scientific_schema=2,
            )

    def test_unknown_context_field_is_rejected_prelaunch(self):
        raw = SCIENTIFIC_SCHEMA2.replace(
            b'    write_gate_result(context, gate_outcomes={"materiality": "favorable"})',
            b'    output_id = context.output_id\n'
            b'    write_gate_result(context, gate_outcomes={"materiality": "favorable"})',
        )
        with self.assertRaisesRegex(READY.ReadyError, "unknown RouteContext field: output_id"):
            READY.check_entrypoint(raw, "program.py", scientific_schema=2)

    def test_complete_units_requires_generic_ledger_calls(self):
        with self.assertRaisesRegex(READY.ReadyError, "completed-unit ledger"):
            READY.check_entrypoint(
                SCIENTIFIC_SCHEMA2, "program.py", scientific_schema=2,
                require_unit_ledger=True,
            )

    def test_schema2_unknown_engineering_field_is_rejected_prelaunch(self):
        raw = SCHEMA2_ENGINEERING.replace(
            b'"scientific_training_occurred": False,',
            b'"scientific_training_occurred": False, "reference_checks": {},',
        )
        with self.assertRaisesRegex(READY.ReadyError, "reference_checks"):
            READY.check_entrypoint(raw, "program.py", require_engineering=True)

    def test_schema2_missing_engineering_payload_is_rejected_prelaunch(self):
        with self.assertRaisesRegex(READY.ReadyError, "requires engineering evidence"):
            READY.check_entrypoint(GOOD, "program.py", require_engineering=True)

    def test_schema2_engineering_dict_mutation_is_rejected_prelaunch(self):
        raw = SCHEMA2_ENGINEERING.replace(
            b'    write_contract_result(context, checks={"ok": True}, engineering=engineering)',
            b'    engineering["reference_checks"] = {}\n'
            b'    write_contract_result(context, checks={"ok": True}, engineering=engineering)',
        )
        with self.assertRaisesRegex(READY.ReadyError, "cannot be mutated"):
            READY.check_entrypoint(raw, "program.py", require_engineering=True)

    def test_positional_output_entrypoint_is_rejected(self):
        with self.assertRaises(READY.ReadyError):
            READY.check_entrypoint(b"def run(output_dir):\n    return output_dir\n", "program.py")

    def test_names_without_required_calls_are_rejected(self):
        raw = b'''\
def contract(context_path):
    return "load_context write_contract_result"
def run(context_path):
    return "load_context write_run_result"
def main():
    return "contract run --context"
'''
        with self.assertRaises(READY.ReadyError):
            READY.check_entrypoint(raw, "program.py")

    def test_route_wide_published_names_are_write_once(self):
        owners = {}
        READY.claim_published_name(owners, "summary.json", "S0 evidence")
        with self.assertRaises(READY.ReadyError):
            READY.claim_published_name(owners, "summary.json", "D0 evidence")

    def test_common_authoring_errors_are_reported_together(self):
        manifest = {
            "operations": {
                "S0": {
                    "monitor_profile": ["fast"],
                    "allowed_terminal_tuples": [
                        {"state": "PASS", "decision": "PASS", "authorizes": "A0"},
                    ],
                },
            },
        }
        errors = READY.authoring_errors(
            manifest, ["--launch-ready requires Status: PLANNED"],
        )
        self.assertEqual(3, len(errors))
        self.assertTrue(any("route card" in error for error in errors))
        self.assertTrue(any("monitor_profile" in error for error in errors))
        self.assertTrue(any("FAILED_ENGINEERING / null / NONE" in error for error in errors))

    def test_valid_common_authoring_fields_add_no_errors(self):
        manifest = {
            "operations": {
                "S0": {
                    "monitor_profile": "short",
                    "allowed_terminal_tuples": [
                        READY.GENERIC_ENGINEERING_TERMINAL.copy(),
                    ],
                },
            },
        }
        self.assertEqual([], READY.authoring_errors(manifest, []))

    def test_evidence_readme_is_not_a_route_ready_dependency(self):
        source = Path(READY.__file__).read_text(encoding="utf-8")
        self.assertNotIn("experiment_logs/{manifest['route_id']}/README.md", source)

    def test_optional_precision_is_initialized_for_not_applicable_contracts(self):
        source = Path(READY.__file__).read_text(encoding="utf-8")
        initialization = source.index("            precision = None\n")
        conditional = source.index("            if precision_path is not None:\n")
        alignment = source.index(
            "                ops.validate_contract_runtime_alignment(contract, spec, precision)\n"
        )
        self.assertLess(initialization, conditional)
        self.assertLess(conditional, alignment)


if __name__ == "__main__":
    unittest.main()
