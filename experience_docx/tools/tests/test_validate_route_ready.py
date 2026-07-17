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


if __name__ == "__main__":
    unittest.main()
