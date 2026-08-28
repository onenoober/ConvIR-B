"""Cloud-only tests for fixed dataset identity and protected access."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import experiment_assistant_datasets as DATASETS  # noqa: E402


def write_registry(root, *, role="locked_test", protected=True, path=None):
    training = root / "training"
    sealed = root / "sealed"
    training.mkdir(exist_ok=True)
    sealed.mkdir(exist_ok=True)
    registry_path = root / "registry.json"
    registry_path.write_text(json.dumps({
        "schema_version": 1,
        "datasets": [
            {
                "id": "train", "role": "training", "path": str(training),
                "identity_sha256": "1" * 64, "protected": False,
            },
            {
                "id": "sealed", "role": role,
                "path": str(sealed if path is None else path),
                "identity_sha256": "2" * 64, "protected": protected,
            },
        ],
    }) + "\n", encoding="utf-8")
    return registry_path


class ExperimentAssistantDatasetTests(unittest.TestCase):
    def test_unprotected_resolution_is_identity_bound_without_public_path(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            registry = DATASETS.DatasetRegistry(write_registry(root))
            resolved = registry.resolve(
                [{"id": "train", "role": "training"}], [],
            )
            self.assertEqual(64, len(resolved["bindings_sha256"]))
            self.assertIn("path", resolved["bindings"][0])
            self.assertNotIn("path", resolved["public_bindings"][0])

    def test_protected_role_denies_by_default_and_explicit_role_allows(self):
        with tempfile.TemporaryDirectory() as raw:
            registry = DATASETS.DatasetRegistry(write_registry(Path(raw)))
            requested = [{"id": "sealed", "role": "locked_test"}]
            with self.assertRaisesRegex(DATASETS.DatasetRegistryError, "explicit"):
                registry.resolve(requested, [])
            allowed = registry.resolve(requested, ["locked_test"])
            self.assertTrue(allowed["public_bindings"][0]["protected"])

    def test_registry_role_conflict_blocks(self):
        with tempfile.TemporaryDirectory() as raw:
            registry = DATASETS.DatasetRegistry(write_registry(Path(raw)))
            with self.assertRaisesRegex(DATASETS.DatasetRegistryError, "role mismatch"):
                registry.resolve([{"id": "train", "role": "test"}], [])

    def test_protected_flag_must_match_protected_role(self):
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(DATASETS.DatasetRegistryError, "protected flag"):
                DATASETS.DatasetRegistry(
                    write_registry(Path(raw), role="locked_test", protected=False)
                )

    def test_dataset_symlink_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            outside = root / "outside"
            outside.mkdir()
            linked = root / "linked"
            os.symlink(outside, linked)
            with self.assertRaisesRegex(DATASETS.DatasetRegistryError, "regular file or directory"):
                DATASETS.DatasetRegistry(write_registry(root, path=linked))

    def test_registry_byte_change_is_diagnostic_when_dataset_bindings_are_identical(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = write_registry(root)
            first = DATASETS.DatasetRegistry(path)
            first_resolution = first.resolve(
                [{"id": "train", "role": "training"}], [],
            )
            value = json.loads(path.read_text(encoding="utf-8"))
            path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
            second = DATASETS.DatasetRegistry(path)
            second_resolution = second.resolve(
                [{"id": "train", "role": "training"}], [],
            )
            self.assertNotEqual(first.registry_sha256, second.registry_sha256)
            self.assertEqual(
                first_resolution["bindings_sha256"],
                second_resolution["bindings_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
