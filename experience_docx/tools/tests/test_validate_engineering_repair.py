"""Tests for conservative same-contract engineering repair classification."""

import ast
import unittest

import validate_engineering_repair as REPAIR


class EngineeringRepairTests(unittest.TestCase):
    def test_symbol_qualification_and_contract_fixture_are_safe(self):
        before = b'''\
import old_module
LIMIT = 3
def contract(context):
    return True
def run(value):
    return old_module.clamp(value, LIMIT)
'''
        after = b'''\
from fixed_module import clamp
LIMIT = 3
def contract(context):
    assert clamp(4, 3) == 3
    return True
def run(value):
    return clamp(value, LIMIT)
'''
        self.assertEqual(
            REPAIR.normalized_entrypoint(before),
            REPAIR.normalized_entrypoint(after),
        )

    def test_algorithm_constant_change_is_sensitive(self):
        before = b"LIMIT=3\ndef run(x):\n return min(x, LIMIT)\n"
        after = b"LIMIT=4\ndef run(x):\n return min(x, LIMIT)\n"
        self.assertNotEqual(
            REPAIR.normalized_entrypoint(before),
            REPAIR.normalized_entrypoint(after),
        )

    def test_new_algorithm_helper_is_sensitive(self):
        before = b"def run(x):\n return x\n"
        after = b"def helper(x):\n return x * 2\ndef run(x):\n return helper(x)\n"
        self.assertNotEqual(
            REPAIR.normalized_entrypoint(before),
            REPAIR.normalized_entrypoint(after),
        )

    def test_same_identity_file_path_change_is_safe(self):
        base = {"schema_version": 1, "route_id": "r", "operation_id": "A0", "assets": [{
            "id": "manifest", "kind": "file", "path": "/old", "sha256": "a" * 64,
            "access_role": "development_screening", "contract_access": False,
        }]}
        candidate = {**base, "assets": [{**base["assets"][0], "path": "/new"}]}
        self.assertEqual(["manifest"], REPAIR.validate_asset_repair(base, candidate))

    def test_directory_path_change_is_sensitive(self):
        base = {"schema_version": 1, "route_id": "r", "operation_id": "A0", "assets": [{
            "id": "data", "kind": "directory", "path": "/old",
            "access_role": "development_screening", "contract_access": False,
        }]}
        candidate = {**base, "assets": [{**base["assets"][0], "path": "/new"}]}
        with self.assertRaises(REPAIR.RepairError):
            REPAIR.validate_asset_repair(base, candidate)

    def test_card_allows_output_and_standard_repair_note_only(self):
        before = b"- output: run-r1\n- metric: fixed\n"
        after = b"- output: run-r2\n- metric: fixed\n- Same-contract engineering repair: path only\n"
        self.assertEqual(
            REPAIR.normalize_card(before, "run-r1", "run-r1"),
            REPAIR.normalize_card(after, "run-r1", "run-r2"),
        )


if __name__ == "__main__":
    unittest.main()
