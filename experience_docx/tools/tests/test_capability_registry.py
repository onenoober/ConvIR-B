"""Tests for exact-identity engineering capability reuse."""

import copy
import json
import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import capability_registry as REGISTRY  # noqa: E402


def identity():
    return {
        "source_commit": "a" * 40,
        "code_path_sha256": "b" * 64,
        "checkpoint_sha256": "c" * 64,
        "runtime_environment_sha256": "d" * 64,
        "device_class": "cuda_sm89",
        "input_contract_sha256": "e" * 64,
    }


def record():
    value = {
        "schema_version": 1,
        "qualification_id": "qualification_1",
        "identity": identity(),
        "identity_sha256": REGISTRY.identity_digest(identity()),
        "status": "PASSED_ENGINEERING",
        "contract_mode": "gpu_synthetic_no_data",
        "evidence_relpath": "experience_docx/experiment_logs/qualification/closeout.json",
        "evidence_sha256": "f" * 64,
        "scientific_authorization": "NONE",
        "protected_data_touched": False,
    }
    return value


class CapabilityRegistryTests(unittest.TestCase):
    def test_exact_identity_match_reuses_engineering_only(self):
        records = REGISTRY.load_records(
            [json.dumps(record())], evidence_exists=lambda _: True,
        )
        result = REGISTRY.lookup(records, identity())
        self.assertEqual("CAPABILITY_REUSE_EXACT_MATCH", result["status"])
        self.assertTrue(result["engineering_reuse_authorized"])
        self.assertEqual("NONE", result["scientific_authorization"])

    def test_each_identity_field_change_invalidates_reuse(self):
        records = REGISTRY.load_records([json.dumps(record())])
        replacements = {
            "source_commit": "0" * 40,
            "code_path_sha256": "0" * 64,
            "checkpoint_sha256": "0" * 64,
            "runtime_environment_sha256": "0" * 64,
            "device_class": "cuda_sm80",
            "input_contract_sha256": "0" * 64,
        }
        for field, replacement in replacements.items():
            candidate = identity()
            candidate[field] = replacement
            result = REGISTRY.lookup(records, candidate)
            self.assertEqual("CAPABILITY_REUSE_MISS", result["status"], field)
            self.assertFalse(result["engineering_reuse_authorized"], field)

    def test_scientific_authorization_or_protected_touch_is_never_reusable(self):
        scientific = record()
        scientific["scientific_authorization"] = "PROMOTION"
        with self.assertRaises(REGISTRY.CapabilityRegistryError):
            REGISTRY.validate_record(scientific)
        protected = record()
        protected["protected_data_touched"] = True
        with self.assertRaises(REGISTRY.CapabilityRegistryError):
            REGISTRY.validate_record(protected)

    def test_missing_evidence_bad_digest_and_nonpass_status_are_rejected(self):
        with self.assertRaises(REGISTRY.CapabilityRegistryError):
            REGISTRY.validate_record(record(), evidence_exists=lambda _: False)
        with self.assertRaises(REGISTRY.CapabilityRegistryError):
            REGISTRY.validate_record(record(), read_evidence=lambda _: b"tampered")
        bad = record()
        bad["identity_sha256"] = "0" * 64
        with self.assertRaises(REGISTRY.CapabilityRegistryError):
            REGISTRY.validate_record(bad)
        failed = record()
        failed["status"] = "FAILED_ENGINEERING"
        with self.assertRaises(REGISTRY.CapabilityRegistryError):
            REGISTRY.validate_record(failed)

    def test_duplicate_identity_or_qualification_is_rejected(self):
        duplicate = copy.deepcopy(record())
        with self.assertRaises(REGISTRY.CapabilityRegistryError):
            REGISTRY.load_records([json.dumps(record()), json.dumps(duplicate)])
        duplicate["qualification_id"] = "qualification_2"
        with self.assertRaises(REGISTRY.CapabilityRegistryError):
            REGISTRY.load_records([json.dumps(record()), json.dumps(duplicate)])

    def test_point_lookup_ignores_unrelated_history_but_validates_match(self):
        unrelated = {"identity_sha256": "0" * 64, "historical": "malformed"}
        result = REGISTRY.lookup_lines(
            ["not-json", json.dumps(unrelated), json.dumps(record())],
            identity(), evidence_exists=lambda _: True,
        )
        self.assertEqual("CAPABILITY_REUSE_EXACT_MATCH", result["status"])
        broken_match = record()
        broken_match["evidence_sha256"] = "invalid"
        with self.assertRaises(REGISTRY.CapabilityRegistryError):
            REGISTRY.lookup_lines([json.dumps(broken_match)], identity())

    def test_point_lookup_miss_does_not_validate_unrelated_evidence(self):
        candidate = identity()
        candidate["device_class"] = "cuda_sm80"
        result = REGISTRY.lookup_lines(
            ["not-json", json.dumps(record())], candidate,
            evidence_exists=lambda _: (_ for _ in ()).throw(AssertionError()),
        )
        self.assertEqual("CAPABILITY_REUSE_MISS", result["status"])


if __name__ == "__main__":
    unittest.main()
