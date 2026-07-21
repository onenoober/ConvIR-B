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


class RouteReadyTests(unittest.TestCase):
    def test_standard_entrypoint_interface_passes(self):
        READY.check_entrypoint(GOOD, "experience_docx/tools/program.py")

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
