"""Tests for safe typed route asset manifest generation."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import build_route_asset_manifest as BUILDER  # noqa: E402


def spec():
    return {
        "route_id": "route", "operation_id": "A0",
        "protected_data_permissions": {
            "allow_confirmation": False, "allow_canary": False,
            "allow_locked_test": False,
        },
    }


def full_manifest_and_spec():
    operation = {
        "runner_relpath": "experience_docx/tools/run_route_operation.sh",
        "mode": "a0", "require_gpu": False, "output_id": "a0-r1",
        "closeout_filename": "a0_closeout.json", "prior_closeout_relpath": None,
        "prior_terminal_tuple": None,
        "allowed_terminal_tuples": [
            {"state": "COMPLETED_GATE_PASS", "decision": "PASS", "authorizes": "NONE"},
            {"state": "FAILED_ENGINEERING", "decision": None, "authorizes": "NONE"},
        ],
        "workspace_policy": "fresh_route", "output_policy": "new",
        "monitor_profile": "short", "heartbeat_timeout_seconds": 120,
        "min_free_gpu_mib": 0, "max_gpu_utilization_pct": 100,
    }
    manifest = {"route_id": "route", "operations": {"A0": operation}}
    runtime = {
        "schema_version": 1, "route_id": "route", "operation_id": "A0",
        "entrypoint_relpath": "experience_docx/tools/a0.py",
        "asset_manifest_relpath": "experience_docx/route_assets/A0.json",
        "timeout_seconds": 600, "expected_wall_seconds": 60, "total_units": 1,
        "evidence_role": "engineering_debug", "resume_policy": "none",
        "protected_data_permissions": {
            "allow_confirmation": False, "allow_canary": False,
            "allow_locked_test": False,
        },
        "environment": {},
        "evidence_files": [{
            "source_relpath": "workload/summary.json",
            "destination_filename": "summary.json", "required": True,
            "max_bytes": 4096,
        }],
    }
    return manifest, runtime


class AssetManifestBuilderTests(unittest.TestCase):
    def test_unrestricted_local_file_is_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.txt"
            path.write_bytes(b"asset")
            request = {
                "schema_version": 1,
                "assets": [{
                    "id": "metadata", "kind": "file", "path": "/remote/asset.txt",
                    "access_role": "unrestricted", "contract_access": True,
                    "identity": {"local_file": str(path)},
                }],
            }
            manifest = BUILDER.build_manifest(spec(), request, 1024)
            self.assertEqual(
                hashlib.sha256(b"asset").hexdigest(), manifest["assets"][0]["sha256"],
            )

    def test_development_asset_requires_predeclared_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.txt"
            path.write_bytes(b"protected")
            request = {
                "schema_version": 1,
                "assets": [{
                    "id": "development", "kind": "file", "path": "/remote/data.bin",
                    "access_role": "development_screening", "contract_access": False,
                    "identity": {"local_file": str(path)},
                }],
            }
            with self.assertRaises(BUILDER.AssetBuildError):
                BUILDER.build_manifest(spec(), request, 1024)
            request["assets"][0]["identity"] = {"sha256": "a" * 64}
            manifest = BUILDER.build_manifest(spec(), request, 1024)
            self.assertEqual("a" * 64, manifest["assets"][0]["sha256"])

    def test_confirmation_asset_cannot_reach_contract(self):
        request = {
            "schema_version": 1,
            "assets": [{
                "id": "confirmation", "kind": "file", "path": "/remote/confirmation.bin",
                "access_role": "confirmation", "contract_access": True,
                "identity": {"sha256": "b" * 64},
            }],
        }
        with self.assertRaises(BUILDER.AssetBuildError):
            BUILDER.build_manifest(spec(), request, 1024)

    def test_apply_writes_declared_manifest_without_protected_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            spec_path = repo / "experience_docx/route_runtime_specs/A0.json"
            manifest_path = repo / "experience_docx/route_operations.json"
            spec_path.parent.mkdir(parents=True)
            manifest, runtime = full_manifest_and_spec()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            spec_path.write_text(json.dumps(runtime), encoding="utf-8")
            request_path = root / "assets.json"
            request_path.write_text(json.dumps({
                "schema_version": 1,
                "assets": [{
                    "id": "metadata", "kind": "file", "path": "/remote/metadata.json",
                    "access_role": "engineering_debug", "contract_access": True,
                    "identity": {"sha256": "c" * 64},
                }],
            }), encoding="utf-8")
            report = BUILDER.prepare(
                repo, "A0", request_path, maximum_hash_bytes=1024, apply=True,
            )
            self.assertEqual("ASSET_MANIFEST_APPLIED", report["status"])
            self.assertEqual(0, report["local_content_reads"])
            self.assertTrue((repo / report["asset_manifest_relpath"]).is_file())


if __name__ == "__main__":
    unittest.main()
