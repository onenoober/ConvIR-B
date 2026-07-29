#!/usr/bin/env python3
"""Synthetic-only tests for the identity-bound cloud inventory core."""

import hashlib
import json
import os
import subprocess
import tempfile
import tracemalloc
import unittest
from pathlib import Path
from unittest import mock

import convir_evidence_catalog as catalog
import convir_evidence_cloud_inventory as inventory


def raw_json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def file_record(path, raw, *, source_path=None):
    value = {
        "path": path,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if source_path is not None:
        value["source_path"] = source_path
    return value


class SyntheticTerminal:
    route_id = "route-a"
    operation_id = "A0"
    run_id = "route-a-r1"
    route_commit = "b" * 40
    runner_sha256 = "c" * 64
    result_raw = b'{"metric":1}\n'

    def __init__(self, root):
        self.repo = Path(root) / "repo"
        self.repo.mkdir()
        self.git("init", "-b", "main")
        self.git("config", "user.email", "inventory@example.invalid")
        self.git("config", "user.name", "Inventory Test")

    def git(self, *args):
        return subprocess.run(
            ["/usr/bin/git", "-C", str(self.repo), *args],
            capture_output=True, text=True, check=True,
        )

    def build(self, *, terminal_schema=2, output_id=None,
              manifest_closeout=None, closeout_overrides=None,
              conclusion_overrides=None, closeout_evidence_overrides=None,
              evidence_role="development_screening", permissions=None,
              protected_touches=None, tamper_result=False,
              include_unmapped=False, duplicate_runtime_mapping=False,
              duplicate_bundle_source=False,
              include_optional_unarchived=False,
              invalid_runtime_source=False,
              shared_source_mapping=False,
              noncanonical_manifest_archive=False,
              noncanonical_result=False,
              manifest_route_card_mismatch=False):
        output_id = self.run_id if output_id is None else output_id
        closeout_filename = "route_closeout.json"
        manifest_closeout = closeout_filename \
            if manifest_closeout is None else manifest_closeout
        permissions = {
            "allow_confirmation": False,
            "allow_canary": False,
            "allow_locked_test": False,
        } if permissions is None else permissions
        protected_touches = {
            "confirmation_images_targets_outcomes_touched": False,
            "canary_touched": False,
            "locked_test_touched": False,
        } if protected_touches is None else protected_touches
        prefix = f"experience_docx/experiment_logs/{self.route_id}"
        contract_path = f"experience_docx/experiment_cards/{self.route_id}.md"
        closeout_path = f"{prefix}/{closeout_filename}"
        conclusion_path = f"{prefix}/route_conclusion.json"
        result_path = (
            f"{prefix}/nested/metric.json" if noncanonical_result
            else f"{prefix}/metric.json"
        )
        manifest_path = (
            f"{prefix}/launch_contract/{self.operation_id}/nested/manifest.json"
            if noncanonical_manifest_archive else
            f"{prefix}/launch_contract/{self.operation_id}/manifest.json"
        )
        runtime_path = f"{prefix}/launch_contract/{self.operation_id}/runtime_spec.json"
        experiment_spec_source = "experience_docx/experiment_specs/route-a.json"
        program_contract_source = "experience_docx/research_programs/route-a.json"
        scientific_contract_source = (
            "experience_docx/scientific_contracts/route-a__A0.json"
        )
        precision_source = "experience_docx/precision_certificates/route-a__A0.json"
        contract_raw = b"# Route A\n"
        manifest = {
            "schema_version": 6,
            "route_id": self.route_id,
            "operations": {
                self.operation_id: {
                    "output_id": output_id,
                    "closeout_filename": manifest_closeout,
                    "runner_relpath": "experience_docx/tools/run_route_operation.sh",
                    "require_gpu": False,
                    "output_policy": "new",
                    "allowed_terminal_tuples": [{
                        "state": "COMPLETED_GATE_PASS",
                        "decision": "A0_PASS",
                        "authorizes": "NEXT_STAGE",
                    }],
                }
            },
            "route_card_relpath": (
                "experience_docx/experiment_cards/different.md"
                if manifest_route_card_mismatch else contract_path
            ),
            "experiment_spec_relpath": experiment_spec_source,
            "program_contract_relpath": program_contract_source,
            "scientific_contract_relpaths": {
                self.operation_id: scientific_contract_source,
            },
        }
        runtime = {
            "schema_version": 2,
            "route_id": self.route_id,
            "operation_id": self.operation_id,
            "entrypoint_relpath": "experience_docx/tools/qualify_synthetic_inventory.py",
            "asset_manifest_relpath": None,
            "timeout_seconds": 300,
            "expected_wall_seconds": 60,
            "total_units": 1,
            "evidence_role": evidence_role,
            "resume_policy": "none",
            "protected_data_permissions": permissions,
            "environment": {},
            "evidence_files": [{
                "source_relpath": (
                    "control/lifecycle_identity.json" if invalid_runtime_source
                    else "workload/metric.json"
                ),
                "destination_filename": "metric.json",
                "required": True,
                "max_bytes": 4096,
            }],
            "engineering_contract": {
                "mode": "metadata_only",
                "capability_profile_relpath": None,
                "max_seconds": 30,
                "cost_contract": None,
            },
            "precision_contract": {
                "mode": (
                    "descriptive_capacity"
                    if evidence_role == "development_screening"
                    and not any(permissions.values()) else "formal_precision"
                ),
                "certificate_relpath": precision_source,
                "rationale": "Synthetic inventory contract validation only.",
            },
        }
        if duplicate_runtime_mapping:
            runtime["evidence_files"].append(dict(runtime["evidence_files"][0]))
        if include_optional_unarchived:
            runtime["evidence_files"].append({
                "source_relpath": "workload/optional.json",
                "destination_filename": "optional.json",
                "required": False,
                "max_bytes": 4096,
            })
        if shared_source_mapping:
            runtime["evidence_files"].append({
                "source_relpath": "workload/metric.json",
                "destination_filename": "metric-copy.json",
                "required": True,
                "max_bytes": 4096,
            })
        manifest_raw = raw_json(manifest)
        runtime_raw = raw_json(runtime)
        result_files = [file_record(result_path, self.result_raw)]
        extra_files = {}
        if shared_source_mapping:
            copy_path = f"{prefix}/metric-copy.json"
            result_files.append(file_record(copy_path, self.result_raw))
            extra_files[copy_path] = self.result_raw
        if include_unmapped:
            unmapped_path = f"{prefix}/unmapped.json"
            unmapped_raw = b'{"unmapped":true}\n'
            result_files.append(file_record(unmapped_path, unmapped_raw))
            extra_files[unmapped_path] = unmapped_raw
        closeout = {
            "schema_version": 2,
            "route_id": self.route_id,
            "operation_id": self.operation_id,
            "run_id": self.run_id,
            "route_commit": self.route_commit,
            "runner_sha256": self.runner_sha256,
            "state": "COMPLETED_GATE_PASS",
            "decision": "A0_PASS",
            "authorizes": "NEXT_STAGE",
            "evidence_role": evidence_role,
            "evidence_sha256": {
                Path(item["path"]).name: item["sha256"] for item in result_files
            },
            **protected_touches,
        }
        closeout["evidence_sha256"].update(closeout_evidence_overrides or {})
        closeout.update(closeout_overrides or {})
        closeout_raw = raw_json(closeout)
        conclusion = {
            "schema_version": 1,
            "route_id": self.route_id,
            "operation_id": self.operation_id,
            "run_id": self.run_id,
            "state": "COMPLETED_GATE_PASS",
            "decision": "A0_PASS",
            "authorizes": "NEXT_STAGE",
            "primary_result": "synthetic",
        }
        conclusion.update(conclusion_overrides or {})
        conclusion_raw = raw_json(conclusion)
        record = {
            "schema_version": terminal_schema,
            "route_id": self.route_id,
            "operation_id": self.operation_id,
            "run_id": self.run_id,
            "state": "COMPLETED_GATE_PASS",
            "decision": "A0_PASS",
            "authorizes": "NEXT_STAGE",
            "receipt": "a" * 64,
            "route_commit": self.route_commit,
            "contract_path": contract_path,
            "closeout_path": closeout_path,
            "conclusion_path": conclusion_path,
            "result_paths": [item["path"] for item in result_files],
        }
        bundle_files = {}
        if terminal_schema == 2:
            launch_root = f"{prefix}/launch_contract/{self.operation_id}"
            bundle_values = [
                (manifest_path, inventory.MANIFEST_SOURCE_PATH, manifest_raw),
                (f"{launch_root}/route_note.md", contract_path, contract_raw),
                (
                    f"{launch_root}/experiment_spec.json",
                    experiment_spec_source, b'{"schema_version":1}\n',
                ),
                (
                    f"{launch_root}/program_contract.json",
                    program_contract_source, b'{"schema_version":1}\n',
                ),
                (
                    f"{launch_root}/scientific_contract.json",
                    scientific_contract_source, b'{"schema_version":1}\n',
                ),
                (
                    runtime_path,
                    f"{inventory.RUNTIME_SPEC_PREFIX}{self.operation_id}.json",
                    runtime_raw,
                ),
                (
                    f"{launch_root}/precision_certificate.json",
                    precision_source, b'{"schema_version":1}\n',
                ),
            ]
            contract_bundle = [
                file_record(path, raw, source_path=source)
                for path, source, raw in bundle_values
            ]
            bundle_files = {path: raw for path, _, raw in bundle_values}
            if duplicate_bundle_source:
                duplicate_path = (
                    f"{prefix}/launch_contract/{self.operation_id}/manifest-copy.json"
                )
                contract_bundle.append(file_record(
                    duplicate_path, manifest_raw,
                    source_path=inventory.MANIFEST_SOURCE_PATH,
                ))
                bundle_files[duplicate_path] = manifest_raw
            record.update({
                "contract_bundle": contract_bundle,
                "prior_terminal_record": {
                    "prior_closeout_path": None,
                    "prior_terminal_tuple": None,
                },
                "result_files": result_files,
                "contract_sha256": hashlib.sha256(contract_raw).hexdigest(),
                "closeout_sha256": hashlib.sha256(closeout_raw).hexdigest(),
                "conclusion_sha256": hashlib.sha256(conclusion_raw).hexdigest(),
            })
        files = {
            contract_path: contract_raw,
            closeout_path: closeout_raw,
            conclusion_path: conclusion_raw,
            result_path: (
                self.result_raw + b"tampered" if tamper_result else self.result_raw
            ),
            **bundle_files,
            **extra_files,
        }
        for relative, raw in files.items():
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        line = raw_json(record)
        index = self.repo / catalog.INDEX_PATH
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_bytes(line)
        self.git("add", "--", "experience_docx")
        self.git("commit", "-m", "snapshot")
        commit = self.git("rev-parse", "HEAD").stdout.strip()
        terminal_sha = hashlib.sha256(line.rstrip(b"\n")).hexdigest()
        loaded = catalog.load_catalog(self.repo, commit)
        return {
            "commit": commit,
            "catalog_sha256": loaded["catalog_sha256"],
            "terminal_record_sha256": terminal_sha,
            "record": record,
        }

    def prepare(self, snapshot):
        return inventory.prepare_terminal_binding(
            self.repo,
            snapshot["commit"],
            snapshot["catalog_sha256"],
            snapshot["terminal_record_sha256"],
        )

    def run_root(self, root, binding, *, result=True, extra=0):
        run_root = Path(root) / "synthetic-run"
        (run_root / "control").mkdir(parents=True)
        (run_root / "workload").mkdir()
        (run_root / "control/lifecycle_identity.json").write_bytes(
            raw_json(binding["expected_lifecycle_identity"])
        )
        if result:
            (run_root / "workload/metric.json").write_bytes(self.result_raw)
        for index in range(extra):
            (run_root / f"raw-{index:05d}.bin").touch()
        return run_root


class EvidenceCloudInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="convir cloud inventory ")
        self.fixture = SyntheticTerminal(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_schema2_binding_verifies_blobs_and_derives_exact_run_root(self):
        snapshot = self.fixture.build()
        binding = self.fixture.prepare(snapshot)
        self.assertTrue(binding["eligible"])
        self.assertEqual(self.fixture.run_id, binding["output_id"])
        self.assertEqual(
            f"{inventory.REMOTE_RUNS}/{self.fixture.route_id}/{self.fixture.run_id}",
            binding["run_root"],
        )
        self.assertEqual(1, len(binding["expected_evidence"]))
        self.assertEqual("workload/metric.json", binding["expected_evidence"][0]["source_relpath"])

        loaded = catalog.load_catalog(self.fixture.repo, snapshot["commit"])
        public = loaded["entries"][0]["routes"][0]["terminals"][0]
        self.assertNotIn("contract_bundle", public)
        self.assertNotIn("result_files", public)
        _, records, _ = catalog.load_terminal_records(
            self.fixture.repo, snapshot["commit"]
        )
        self.assertEqual(7, len(records[0]["contract_bundle"]))
        self.assertEqual(1, len(records[0]["result_files"]))

    def test_one_runtime_source_may_map_to_multiple_archived_destinations(self):
        snapshot = self.fixture.build(shared_source_mapping=True)
        binding = self.fixture.prepare(snapshot)
        self.assertEqual(2, len(binding["expected_evidence"]))
        self.assertEqual(
            {"workload/metric.json"},
            {item["source_relpath"] for item in binding["expected_evidence"]},
        )
        run_root = self.fixture.run_root(self.temp.name, binding)
        result = inventory._scan_adapter_root(binding, run_root)
        self.assertEqual("INVENTORY_READY", result["state"])
        self.assertEqual(2, result["reconciliation_counts"]["MATCHED"])

    def test_legacy_unknown_and_protected_terminals_are_not_inventoried(self):
        legacy = self.fixture.build(terminal_schema=1)
        binding = self.fixture.prepare(legacy)
        self.assertFalse(binding["eligible"])
        self.assertEqual("NOT_INVENTORIED", binding["state"])

        unknown = inventory.prepare_terminal_binding(
            self.fixture.repo, legacy["commit"], legacy["catalog_sha256"], "f" * 64
        )
        self.assertEqual("NOT_INVENTORIED", unknown["state"])

        with tempfile.TemporaryDirectory(prefix="convir optional evidence ") as root:
            fixture = SyntheticTerminal(root)
            snapshot = fixture.build(include_optional_unarchived=True)
            optional_binding = fixture.prepare(snapshot)
            self.assertEqual(1, len(optional_binding["optional_evidence"]))
            run_root = fixture.run_root(root, optional_binding)
            (run_root / "workload/optional.json").write_bytes(b'{}\n')
            optional = inventory._scan_adapter_root(optional_binding, run_root)
            self.assertEqual("INVENTORY_READY", optional["state"])
            self.assertEqual(1, optional["policy_counts"][
                "optional_runtime_evidence_without_archive"
            ])
            self.assertEqual(2, optional["reconciliation_counts"]["CLOUD_ONLY"])

    def test_manifest_and_closeout_cross_bindings_fail_closed(self):
        cases = [
            {"output_id": "different-output"},
            {"manifest_closeout": "different-closeout.json"},
            {"closeout_overrides": {"run_id": "different-run"}},
            {"closeout_overrides": {"route_commit": "d" * 40}},
            {"closeout_overrides": {"runner_sha256": "invalid"}},
            {"conclusion_overrides": {"run_id": "different-run"}},
            {"closeout_evidence_overrides": {"unexpected.json": "e" * 64}},
            {"duplicate_runtime_mapping": True},
            {"duplicate_bundle_source": True},
            {"invalid_runtime_source": True},
            {"noncanonical_manifest_archive": True},
            {"noncanonical_result": True},
            {"manifest_route_card_mismatch": True},
        ]
        for index, kwargs in enumerate(cases):
            with self.subTest(case=index):
                with tempfile.TemporaryDirectory(
                    prefix="convir binding conflict "
                ) as root:
                    fixture = SyntheticTerminal(root)
                    snapshot = fixture.build(**kwargs)
                    with self.assertRaises(inventory.InventoryError) as caught:
                        fixture.prepare(snapshot)
                    self.assertEqual("IDENTITY_CONFLICT", caught.exception.state)

    def test_github_blob_tamper_is_detected(self):
        snapshot = self.fixture.build(tamper_result=True)
        with self.assertRaises(inventory.InventoryError) as caught:
            self.fixture.prepare(snapshot)
        self.assertEqual("IDENTITY_CONFLICT", caught.exception.state)

        with mock.patch.object(inventory, "MAX_GITHUB_BOUND_BYTES", 1):
            with self.assertRaises(inventory.InventoryError) as caught:
                self.fixture.prepare(snapshot)
        self.assertEqual("IDENTITY_CONFLICT", caught.exception.state)
        self.assertIn("aggregate bound", str(caught.exception))

    def test_complete_scan_reconciles_matched_and_cloud_only_without_raw_reads(self):
        snapshot = self.fixture.build()
        binding = self.fixture.prepare(snapshot)
        run_root = self.fixture.run_root(self.temp.name, binding, extra=2)
        original = inventory._read_relative_file
        observed = []

        def tracked(root_fd, relative, **kwargs):
            observed.append(relative)
            return original(root_fd, relative, **kwargs)

        with mock.patch.object(
            inventory, "_read_relative_file", side_effect=tracked
        ):
            result = inventory._scan_adapter_root(binding, run_root)
        self.assertEqual("INVENTORY_READY", result["state"])
        self.assertEqual("complete", result["discovery_completeness"])
        self.assertEqual(1, result["reconciliation_counts"]["MATCHED"])
        self.assertEqual(3, result["reconciliation_counts"]["CLOUD_ONLY"])
        self.assertEqual(["workload/metric.json"], observed)
        self.assertNotIn("raw-00000.bin", observed)
        self.assertEqual("adapter_owned_root", result["scope"])
        self.assertFalse(result["root_binding_enforced"])
        self.assertEqual(binding["run_root"], result["declared_run_root"])
        summary = inventory.inventory_summary(result)
        self.assertNotIn("_entries", summary)
        self.assertEqual(result["inventory_sha256"], summary["inventory_sha256"])

    def test_missing_mismatch_unavailable_and_absent_root_are_distinct(self):
        snapshot = self.fixture.build()
        binding = self.fixture.prepare(snapshot)
        run_root = self.fixture.run_root(self.temp.name, binding, result=False)
        missing = inventory._scan_adapter_root(binding, run_root)
        self.assertEqual(1, missing["reconciliation_counts"]["GITHUB_ONLY"])

        (run_root / "workload/metric.json").write_bytes(b"different")
        mismatch = inventory._scan_adapter_root(binding, run_root)
        self.assertEqual("IDENTITY_CONFLICT", mismatch["state"])
        self.assertFalse(mismatch["ok"])
        self.assertEqual(3, mismatch["exit_code"])
        self.assertEqual(1, mismatch["reconciliation_counts"]["IDENTITY_CONFLICT"])

        unavailable = inventory._scan_adapter_root(
            binding, run_root, cloud_available=False
        )
        self.assertEqual("CLOUD_UNAVAILABLE", unavailable["state"])
        self.assertFalse(unavailable["ok"])
        self.assertEqual(3, unavailable["exit_code"])
        self.assertEqual(1, unavailable["reconciliation_counts"]["CLOUD_UNAVAILABLE"])

        absent = inventory._scan_adapter_root(
            binding, Path(self.temp.name) / "does-not-exist"
        )
        self.assertEqual("complete", absent["discovery_completeness"])
        self.assertEqual(1, absent["reconciliation_counts"]["GITHUB_ONLY"])

    def test_expected_path_and_ancestor_type_conflicts_are_not_missing(self):
        snapshot = self.fixture.build()
        binding = self.fixture.prepare(snapshot)

        exact_root = self.fixture.run_root(
            Path(self.temp.name) / "exact-directory", binding, result=False
        )
        (exact_root / "workload/metric.json").mkdir()
        exact = inventory._scan_adapter_root(binding, exact_root)
        self.assertEqual("IDENTITY_CONFLICT", exact["state"])
        self.assertEqual(1, exact["reconciliation_counts"]["IDENTITY_CONFLICT"])
        self.assertEqual(0, exact["reconciliation_counts"]["GITHUB_ONLY"])

        ancestor_root = self.fixture.run_root(
            Path(self.temp.name) / "ancestor-file", binding, result=False
        )
        (ancestor_root / "workload").rmdir()
        (ancestor_root / "workload").write_bytes(b"not-a-directory")
        ancestor = inventory._scan_adapter_root(binding, ancestor_root)
        self.assertEqual("IDENTITY_CONFLICT", ancestor["state"])
        self.assertEqual(1, ancestor["reconciliation_counts"]["IDENTITY_CONFLICT"])
        self.assertEqual(0, ancestor["reconciliation_counts"]["GITHUB_ONLY"])

    def test_lifecycle_identity_and_path_chain_must_not_redirect(self):
        snapshot = self.fixture.build()
        binding = self.fixture.prepare(snapshot)
        for key in binding["expected_lifecycle_identity"]:
            with self.subTest(key=key):
                run_root = self.fixture.run_root(
                    Path(self.temp.name) / key, binding
                )
                changed = dict(binding["expected_lifecycle_identity"])
                changed[key] = 2 if key == "schema_version" else "different"
                (run_root / "control/lifecycle_identity.json").write_bytes(raw_json(changed))
                result = inventory._scan_adapter_root(binding, run_root)
                self.assertEqual("IDENTITY_CONFLICT", result["state"])

        symlink_root = self.fixture.run_root(Path(self.temp.name) / "symlink", binding)
        identity = symlink_root / "control/lifecycle_identity.json"
        target = symlink_root / "identity-target.json"
        target.write_bytes(raw_json(binding["expected_lifecycle_identity"]))
        identity.unlink()
        os.symlink(target, identity)
        result = inventory._scan_adapter_root(binding, symlink_root)
        self.assertEqual("IDENTITY_CONFLICT", result["state"])

        control_root = Path(self.temp.name) / "control-link-root"
        control_root.mkdir()
        control_target = Path(self.temp.name) / "control-target"
        control_target.mkdir()
        (control_target / "lifecycle_identity.json").write_bytes(
            raw_json(binding["expected_lifecycle_identity"])
        )
        os.symlink(control_target, control_root / "control")
        result = inventory._scan_adapter_root(binding, control_root)
        self.assertEqual("IDENTITY_CONFLICT", result["state"])
        self.assertTrue(any(
            issue.startswith("LIFECYCLE_IDENTITY_INVALID:")
            for issue in result["issues"]
        ))

        parent_target = Path(self.temp.name) / "parent-target"
        parent_target.mkdir()
        parent_link = Path(self.temp.name) / "parent-link"
        os.symlink(parent_target, parent_link)
        result = inventory._scan_adapter_root(binding, parent_link / "absent")
        self.assertEqual("IDENTITY_CONFLICT", result["state"])
        self.assertEqual(0, result["reconciliation_counts"]["GITHUB_ONLY"])

    def test_parent_directory_swap_cannot_redirect_formal_evidence_read(self):
        snapshot = self.fixture.build()
        binding = self.fixture.prepare(snapshot)
        run_root = self.fixture.run_root(self.temp.name, binding)
        redirect = Path(self.temp.name) / "redirect-workload"
        redirect.mkdir()
        (redirect / "metric.json").write_bytes(self.fixture.result_raw)
        original_walk = inventory._walk_metadata

        def swap_after_scan(*args, **kwargs):
            result = original_walk(*args, **kwargs)
            (run_root / "workload").rename(run_root / "workload-original")
            os.symlink(redirect, run_root / "workload", target_is_directory=True)
            return result

        with mock.patch.object(
            inventory, "_walk_metadata", side_effect=swap_after_scan
        ):
            result = inventory._scan_adapter_root(binding, run_root)
        self.assertEqual("IDENTITY_CONFLICT", result["state"])
        self.assertEqual(0, result["reconciliation_counts"]["MATCHED"])
        self.assertGreaterEqual(
            result["reconciliation_counts"]["IDENTITY_CONFLICT"], 1
        )

    def test_protected_scope_marks_every_expected_file_not_inventoried(self):
        protected_snapshot = self.fixture.build(
            evidence_role="confirmation",
            permissions={
                "allow_confirmation": True,
                "allow_canary": False,
                "allow_locked_test": False,
            },
        )
        protected = self.fixture.prepare(protected_snapshot)
        self.assertFalse(protected["raw_inventory_authorized"])
        result = inventory._scan_adapter_root(
            protected, Path(self.temp.name) / "not-read"
        )
        self.assertEqual("NOT_INVENTORIED", result["state"])
        self.assertEqual(1, result["reconciliation_counts"]["NOT_INVENTORIED"])
        self.assertEqual(1, result["entry_count"])

        with tempfile.TemporaryDirectory(prefix="convir protected touch ") as root:
            fixture = SyntheticTerminal(root)
            touched_snapshot = fixture.build(protected_touches={
                "confirmation_images_targets_outcomes_touched": False,
                "canary_touched": True,
                "locked_test_touched": False,
            })
            touched = fixture.prepare(touched_snapshot)
            self.assertFalse(touched["raw_inventory_authorized"])
            touched_result = inventory._scan_adapter_root(
                touched, Path(root) / "must-not-be-read"
            )
            self.assertEqual("NOT_INVENTORIED", touched_result["state"])

    def test_active_and_partial_scopes_never_claim_missing(self):
        snapshot = self.fixture.build()
        binding = self.fixture.prepare(snapshot)
        run_root = self.fixture.run_root(self.temp.name, binding, result=False, extra=3)
        active = inventory._scan_adapter_root(
            binding, run_root, active_session=True
        )
        self.assertEqual("NOT_INVENTORIED", active["state"])
        partial = inventory._scan_adapter_root(
            binding, run_root, limits={"max_entries": 2}
        )
        self.assertEqual("partial", partial["discovery_completeness"])
        self.assertEqual(0, partial["reconciliation_counts"]["GITHUB_ONLY"])
        self.assertGreater(partial["reconciliation_counts"]["NOT_INVENTORIED"], 0)

    def test_hard_limits_and_cloud_read_failures_remain_typed(self):
        snapshot = self.fixture.build()
        binding = self.fixture.prepare(snapshot)
        run_root = self.fixture.run_root(self.temp.name, binding)
        with self.assertRaises(inventory.InventoryError) as caught:
            inventory._scan_adapter_root(
                binding, run_root,
                limits={"max_entries": inventory.MAX_SCAN_ENTRIES + 1},
            )
        self.assertEqual("ARGUMENTS_INVALID", caught.exception.state)
        self.assertEqual(2, caught.exception.exit_code)

        unavailable_error = inventory.InventoryError(
            "synthetic stale mount", state="CLOUD_UNAVAILABLE", exit_code=3
        )
        with mock.patch.object(
            inventory, "_read_relative_file", side_effect=unavailable_error
        ):
            unavailable = inventory._scan_adapter_root(binding, run_root)
        self.assertEqual("CLOUD_UNAVAILABLE", unavailable["state"])
        self.assertFalse(unavailable["ok"])
        expected = [
            item for item in unavailable["_entries"]
            if item["artifact_class"] == "formal_compact_evidence"
        ]
        self.assertEqual(
            ["CLOUD_UNAVAILABLE"],
            [item["reconciliation_state"] for item in expected],
        )

    def test_internal_symlink_and_unmapped_result_remain_explicit(self):
        snapshot = self.fixture.build(include_unmapped=True)
        binding = self.fixture.prepare(snapshot)
        self.assertEqual(1, len(binding["unmapped_results"]))
        run_root = self.fixture.run_root(self.temp.name, binding)
        os.symlink(run_root / "workload/metric.json", run_root / "raw-link")
        result = inventory._scan_adapter_root(binding, run_root)
        self.assertEqual("IDENTITY_CONFLICT", result["state"])
        self.assertEqual("partial", result["discovery_completeness"])
        self.assertEqual(1, result["reconciliation_counts"]["IDENTITY_CONFLICT"])
        self.assertEqual(1, result["policy_counts"][
            "github_result_without_runtime_source_mapping"
        ])

        special_root = self.fixture.run_root(
            Path(self.temp.name) / "special", binding
        )
        os.mkfifo(special_root / "raw-pipe")
        special = inventory._scan_adapter_root(binding, special_root)
        self.assertEqual("IDENTITY_CONFLICT", special["state"])
        self.assertEqual(1, special["reconciliation_counts"]["IDENTITY_CONFLICT"])

    def test_query_paginates_without_duplicates_and_rejects_drift(self):
        snapshot = self.fixture.build()
        binding = self.fixture.prepare(snapshot)
        run_root = self.fixture.run_root(self.temp.name, binding, extra=35)
        result = inventory._scan_adapter_root(binding, run_root)
        observed = []
        cursor = None
        first_cursor = None
        while True:
            page = inventory.inventory_query(
                result,
                inventory_sha256=result["inventory_sha256"],
                reconciliation_states=["CLOUD_ONLY"],
                cursor=cursor,
                limit=7,
            )
            self.assertLessEqual(
                len(inventory.canonical_bytes(page)), inventory.MAX_QUERY_VALUE_BYTES
            )
            observed.extend(item["relative_path"] for item in page["entries"])
            if page["complete"]:
                break
            cursor = page["next_cursor"]
            first_cursor = first_cursor or cursor
        self.assertEqual(len(observed), len(set(observed)))
        self.assertEqual(36, len(observed))

        with self.assertRaises(inventory.InventoryError) as caught:
            inventory.inventory_query(
                result,
                inventory_sha256=result["inventory_sha256"],
                reconciliation_states=["MATCHED"],
                cursor=first_cursor,
            )
        self.assertEqual("REPO_CURSOR_IDENTITY_MISMATCH", caught.exception.state)
        with self.assertRaises(inventory.InventoryError) as caught:
            inventory.inventory_query(result, inventory_sha256="f" * 64)
        self.assertEqual("INVENTORY_DRIFT", caught.exception.state)

    def test_same_scale_25000_entry_scan_stays_bounded(self):
        snapshot = self.fixture.build()
        binding = self.fixture.prepare(snapshot)
        run_root = self.fixture.run_root(self.temp.name, binding, extra=24_996)
        tracemalloc.start()
        result = inventory._scan_adapter_root(binding, run_root)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertEqual(25_000, result["scan"]["entry_count"])
        self.assertEqual("complete", result["discovery_completeness"])
        self.assertLessEqual(result["scan"]["elapsed_seconds"], inventory.MAX_SCAN_SECONDS)
        self.assertLess(peak, 256 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
