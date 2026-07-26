"""Tests for the complete minimal terminal-science archive fastpath."""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
import prepare_terminal_archive as ARCHIVE  # noqa: E402


class TerminalArchiveTests(unittest.TestCase):
    def test_empty_exclusion_asset_is_valid_text(self):
        self.assertEqual("", ARCHIVE.checked_text(b"", "route_exclusions.txt"))

    def test_other_empty_text_evidence_is_rejected(self):
        with self.assertRaises(ARCHIVE.TerminalArchiveError):
            ARCHIVE.checked_text(b"", "formal_results.txt")

    def git(self, repo: Path, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, text=True, capture_output=True, check=True,
        ).stdout.strip()

    def init_repo(self, path: Path) -> Path:
        path.mkdir()
        self.git(path, "init", "-q")
        self.git(path, "config", "user.name", "test")
        self.git(path, "config", "user.email", "test@example.com")
        return path

    def source(self, root: Path, *, include_results=True, bad_hash=False,
               verdict_only=False, forbidden=False,
               incomplete_conclusion=False,
               canonical=False) -> tuple[Path, str, str, str, str]:
        repo = self.init_repo(root / "source")
        route_id = "route"
        card = "experience_docx/experiment_cards/route.md"
        card_path = repo / card
        card_path.parent.mkdir(parents=True)
        card_path.write_text("# Route\n\n- Route id: route\n", encoding="utf-8")
        self.git(repo, "add", card)
        operation_id = "A1"
        operations = {
            "schema_version": 6 if canonical else 4,
            "route_id": route_id,
            "operations": {operation_id: {}},
        }
        if canonical:
            operations.update({
                "route_card_relpath": card,
                "experiment_spec_relpath": "experience_docx/experiment_specs/route.json",
                "program_contract_relpath": "experience_docx/research_programs/route.json",
                "scientific_contract_relpaths": {
                    operation_id: "experience_docx/scientific_contracts/route__A1.json",
                },
            })
        manifest = repo / "experience_docx/route_operations.json"
        manifest.write_text(json.dumps(operations), encoding="utf-8")
        spec = {
            "route_id": route_id,
            "operation_id": operation_id,
            "evidence_files": [{
                "destination_filename": "formal_results.csv",
                "required": True,
            }],
        }
        if canonical:
            spec.update({
                "asset_manifest_relpath": None,
                "engineering_contract": {"capability_profile_relpath": None},
                "precision_contract": {"certificate_relpath": None},
            })
        spec_path = repo / f"experience_docx/route_runtime_specs/{operation_id}.json"
        spec_path.parent.mkdir(parents=True)
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        generated = []
        if canonical:
            for relpath in (
                "experience_docx/experiment_specs/route.json",
                "experience_docx/research_programs/route.json",
                "experience_docx/scientific_contracts/route__A1.json",
            ):
                path = repo / relpath
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"route_id": route_id}), encoding="utf-8")
                generated.append(relpath)
        self.git(repo, "add", "experience_docx/route_operations.json",
                 f"experience_docx/route_runtime_specs/{operation_id}.json", *generated)
        self.git(repo, "commit", "-qm", "contract")
        commit = self.git(repo, "rev-parse", "HEAD")
        evidence = repo / "experience_docx/experiment_logs/route"
        evidence.mkdir(parents=True)
        filename = "raw_prediction.png" if forbidden else "formal_results.csv"
        result = b"metric,point,lcb95\ngain,0.03,0.02\n"
        expected = hashlib.sha256(result).hexdigest()
        if bad_hash:
            expected = "0" * 64
        closeout = {
            "route_id": route_id,
            "operation_id": operation_id,
            "run_id": "a1-r1",
            "route_commit": commit,
            "state": "COMPLETED_GATE_PASS",
            "decision": "A1_PASS",
            "authorizes": "NONE",
            "evidence_sha256": {} if verdict_only else {filename: expected},
        }
        closeout_rel = "experience_docx/experiment_logs/route/a1_closeout.json"
        (repo / closeout_rel).write_text(json.dumps(closeout), encoding="utf-8")
        conclusion_rel = "experience_docx/experiment_logs/route/a1_conclusion.json"
        conclusion = {
            "route_id": route_id,
            "operation_id": operation_id,
            "run_id": "a1-r1",
            "state": "COMPLETED_GATE_PASS",
            "decision": "A1_PASS",
            "authorizes": "NONE",
            "primary_result": "gain LCB95 passed",
            "gate_reasons": ["gain LCB95 >= threshold"],
            "competing_explanation": "matched control did not explain the gain",
            "limitations": [] if incomplete_conclusion else ["development population only"],
        }
        (repo / conclusion_rel).write_text(json.dumps(conclusion), encoding="utf-8")
        if include_results:
            (evidence / filename).write_bytes(result)
        return repo, commit, closeout_rel, card, conclusion_rel

    def destination(self, root: Path) -> Path:
        repo = self.init_repo(root / "destination")
        (repo / "README.md").write_text("main\n", encoding="utf-8")
        self.git(repo, "add", "README.md")
        self.git(repo, "commit", "-qm", "main")
        self.git(repo, "update-ref", "refs/remotes/github/main", "HEAD")
        return repo

    def audit(self, source: tuple[Path, str, str, str, str]):
        repo, commit, closeout, card, conclusion = source
        return ARCHIVE.audit_source(
            repo, commit, "route", closeout, card, conclusion, "1" * 64,
        )

    def receipt_copy(self, source, destination: Path) -> dict:
        repo, _, closeout, _, _ = source
        evidence = repo / "experience_docx/experiment_logs/route"
        for filename in (Path(closeout).name, "formal_results.csv"):
            shutil.copyfile(evidence / filename, destination / filename)
        return {}

    def test_complete_bundle_is_staged_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            destination = self.destination(root)
            audit = self.audit(source)
            report = ARCHIVE.prepare_destination(destination, "refs/remotes/github/main", audit)
            self.assertEqual("TERMINAL_ARCHIVE_PREPARED", report["status"])
            self.assertEqual(1, report["preserved_result_files"])
            self.assertEqual(0, report["duplicative_document_updates"])
            self.assertEqual(["commit", "push"], report["remaining_operator_steps"])
            staged = self.git(destination, "diff", "--cached", "--name-only").splitlines()
            self.assertIn(ARCHIVE.INDEX_PATH, staged)
            self.assertIn("experience_docx/experiment_logs/route/formal_results.csv", staged)
            self.assertIn("experience_docx/experiment_logs/route/a1_conclusion.json", staged)
            self.assertNotIn("experience_docx/experiment_logs/route/README.md", staged)

    def test_schema6_archive_preserves_full_launch_contract_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root, canonical=True)
            destination = self.destination(root)
            audit = self.audit(source)
            self.assertEqual(2, audit["schema_version"])
            self.assertEqual(6, len(audit["contract_bundle"]))
            report = ARCHIVE.prepare_destination(
                destination, "refs/remotes/github/main", audit,
            )
            staged = set(report["staged_paths"])
            self.assertIn(
                "experience_docx/experiment_logs/route/launch_contract/A1/manifest.json",
                staged,
            )
            record = ARCHIVE.index_record(audit)
            self.assertEqual(2, record["schema_version"])
            self.assertEqual(6, len(record["contract_bundle"]))
            self.assertEqual(
                {"prior_closeout_path": None, "prior_terminal_tuple": None},
                record["prior_terminal_record"],
            )
            self.assertEqual(1, len(record["result_files"]))
            self.assertEqual(64, len(record["closeout_sha256"]))
            self.assertEqual(64, len(record["conclusion_sha256"]))

    def test_archive_registers_new_engineering_capability_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            repo, commit, closeout_rel, _, _ = source
            identity = {
                "source_commit": "a" * 40,
                "code_path_sha256": "b" * 64,
                "checkpoint_sha256": "c" * 64,
                "runtime_environment_sha256": "d" * 64,
                "device_class": "cuda_sm89",
                "input_contract_sha256": "e" * 64,
            }
            identity_sha = ARCHIVE.capability_registry.identity_digest(identity)
            filename = "a1_capability_qualification.json"
            qualification = {
                "schema_version": 1,
                "qualification_id": f"cap_{identity_sha[:24]}",
                "identity": identity,
                "identity_sha256": identity_sha,
                "status": "PASSED_ENGINEERING",
                "contract_mode": "gpu_synthetic_no_data",
                "route_id": "route", "operation_id": "A1", "run_id": "a1-r1",
                "route_commit": commit,
                "engineering_evidence": {"production_path_exercised": True},
                "scientific_authorization": "NONE",
                "protected_data_touched": False,
            }
            evidence = repo / "experience_docx/experiment_logs/route" / filename
            evidence.write_text(json.dumps(qualification), encoding="utf-8")
            closeout_path = repo / closeout_rel
            closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
            closeout["evidence_sha256"][filename] = hashlib.sha256(
                evidence.read_bytes()
            ).hexdigest()
            closeout["capability_qualification"] = {
                "qualification_id": qualification["qualification_id"],
                "identity_sha256": identity_sha,
                "evidence_filename": filename,
                "status": "PASSED_ENGINEERING",
                "scientific_authorization": "NONE",
            }
            closeout_path.write_text(json.dumps(closeout), encoding="utf-8")
            destination = self.destination(root)
            audit = self.audit(source)
            report = ARCHIVE.prepare_destination(
                destination, "refs/remotes/github/main", audit,
            )
            self.assertIn(
                ARCHIVE.capability_registry.REGISTRY_RELPATH,
                report["staged_paths"],
            )
            registry = destination / ARCHIVE.capability_registry.REGISTRY_RELPATH
            records = ARCHIVE.capability_registry.load_records(
                registry.read_text(encoding="utf-8").splitlines(),
            )
            self.assertEqual(1, len(records))

    def test_missing_formal_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(Path(directory), include_results=False)
            with self.assertRaises(ARCHIVE.TerminalArchiveError):
                self.audit(source)

    def test_tampered_formal_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(Path(directory), bad_hash=True)
            with self.assertRaises(ARCHIVE.TerminalArchiveError):
                self.audit(source)

    def test_verdict_only_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(Path(directory), verdict_only=True)
            with self.assertRaises(ARCHIVE.TerminalArchiveError):
                self.audit(source)

    def test_forbidden_binary_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(Path(directory), forbidden=True)
            with self.assertRaises(ARCHIVE.TerminalArchiveError):
                self.audit(source)

    def test_incomplete_conclusion_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(Path(directory), incomplete_conclusion=True)
            with self.assertRaises(ARCHIVE.TerminalArchiveError):
                self.audit(source)

    def test_dirty_destination_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            destination = self.destination(root)
            (destination / "README.md").write_text("dirty\n", encoding="utf-8")
            with self.assertRaises(ARCHIVE.TerminalArchiveError):
                ARCHIVE.prepare_destination(
                    destination, "refs/remotes/github/main", self.audit(source),
                )

    def test_archive_base_ref_cannot_select_a_local_main_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            destination = self.destination(root)
            with self.assertRaises(ARCHIVE.TerminalArchiveError):
                ARCHIVE.prepare_destination(destination, "main", self.audit(source))

    def test_receipt_evidence_overrides_tampered_local_closeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            repo, commit, closeout, card, conclusion = source
            trusted = root / "receipt"
            trusted.mkdir()
            self.receipt_copy(source, trusted)
            local_closeout = repo / closeout
            tampered = json.loads(local_closeout.read_text(encoding="utf-8"))
            tampered["decision"] = "TAMPERED_LOCAL_DECISION"
            local_closeout.write_text(json.dumps(tampered), encoding="utf-8")
            audit = ARCHIVE.audit_source(
                repo, commit, "route", closeout, card, conclusion, "1" * 64,
                evidence_dir_override=trusted,
                conclusion_dir_override=repo / Path(closeout).parent,
            )
            self.assertEqual("A1_PASS", audit["decision"])

    def test_default_cli_fetches_receipt_evidence_when_local_evidence_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            destination = self.destination(root)
            remote = root / "github.git"
            subprocess.run(["git", "init", "--bare", "-q", remote], check=True)
            self.git(destination, "remote", "add", "github", str(remote))
            self.git(destination, "push", "-q", "github", "HEAD:main")
            repo, commit, closeout, card, conclusion = source
            trusted = root / "receipt"
            trusted.mkdir()
            self.receipt_copy(source, trusted)
            local_closeout = repo / closeout
            tampered = json.loads(local_closeout.read_text(encoding="utf-8"))
            tampered["decision"] = "TAMPERED_LOCAL_DECISION"
            local_closeout.write_text(json.dumps(tampered), encoding="utf-8")

            def fetch(receipt, evidence_dir):
                self.assertEqual("1" * 64, receipt)
                for filename in (Path(closeout).name, "formal_results.csv"):
                    shutil.copyfile(trusted / filename, evidence_dir / filename)
                return {}

            arguments = [
                "prepare_terminal_archive.py",
                "--source-repo", str(repo), "--source-ref", commit,
                "--route-id", "route", "--closeout", closeout,
                "--contract", card, "--conclusion", conclusion,
                "--receipt", "1" * 64, "--destination-repo", str(destination),
            ]
            with mock.patch.object(ARCHIVE, "fetch_receipt_evidence", side_effect=fetch) as mocked:
                with mock.patch.object(sys, "argv", arguments):
                    ARCHIVE.main()
            mocked.assert_called_once()
            index = (destination / ARCHIVE.INDEX_PATH).read_text(encoding="utf-8")
            self.assertIn("A1_PASS", index)
            self.assertNotIn("TAMPERED_LOCAL_DECISION", index)

    def test_conflicting_terminal_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            destination = self.destination(root)
            audit = self.audit(source)
            record = ARCHIVE.index_record(audit)
            record["decision"] = "DIFFERENT"
            index = destination / ARCHIVE.INDEX_PATH
            index.parent.mkdir(parents=True)
            index.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self.git(destination, "add", ARCHIVE.INDEX_PATH)
            self.git(destination, "commit", "-qm", "existing record")
            self.git(destination, "update-ref", "refs/remotes/github/main", "HEAD")
            with self.assertRaises(ARCHIVE.TerminalArchiveError):
                ARCHIVE.prepare_destination(
                    destination, "refs/remotes/github/main", audit,
                )

    def test_finalize_commits_pushes_and_verifies_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            destination = self.destination(root)
            remote = root / "github.git"
            subprocess.run(["git", "init", "--bare", "-q", remote], check=True)
            self.git(destination, "remote", "add", "github", str(remote))
            self.git(destination, "push", "-q", "github", "HEAD:main")
            audit = self.audit(source)
            prepared = ARCHIVE.prepare_destination(
                destination, "refs/remotes/github/main", audit,
            )
            final = ARCHIVE.finalize_destination(
                destination, "route", prepared["staged_paths"], audit,
            )
            self.assertEqual("TERMINAL_ARCHIVE_PUSHED", final["status"])
            self.assertEqual(final["evidence_commit"], final["remote_commit"])
            self.assertEqual([], final["remaining_operator_steps"])
            self.assertEqual("", self.git(destination, "status", "--porcelain"))

    def test_finalize_recovers_one_concurrent_main_fast_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            destination = self.destination(root)
            remote = root / "github.git"
            subprocess.run(["git", "init", "--bare", "-q", remote], check=True)
            self.git(destination, "remote", "add", "github", str(remote))
            self.git(destination, "push", "-q", "github", "HEAD:main")
            audit = self.audit(source)
            prepared = ARCHIVE.prepare_destination(
                destination, "refs/remotes/github/main", audit,
            )
            concurrent = root / "concurrent"
            subprocess.run(
                ["git", "clone", "-q", "--branch", "main", str(remote), concurrent],
                check=True,
            )
            self.git(concurrent, "config", "user.name", "test")
            self.git(concurrent, "config", "user.email", "test@example.com")
            (concurrent / "CONCURRENT.md").write_text("advance\n", encoding="utf-8")
            self.git(concurrent, "add", "CONCURRENT.md")
            self.git(concurrent, "commit", "-qm", "concurrent advance")
            self.git(concurrent, "push", "-q", "origin", "main")
            final = ARCHIVE.finalize_destination(
                destination, "route", prepared["staged_paths"], audit,
            )
            self.assertTrue(final["concurrent_main_advance_recovered"])
            self.assertEqual(final["evidence_commit"], final["remote_commit"])
            self.assertEqual("", self.git(destination, "status", "--porcelain"))

    def test_cli_commit_and_push_is_single_archive_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            destination = self.destination(root)
            remote = root / "github.git"
            subprocess.run(["git", "init", "--bare", "-q", remote], check=True)
            self.git(destination, "remote", "add", "github", str(remote))
            self.git(destination, "push", "-q", "github", "HEAD:main")
            repo, commit, closeout, card, conclusion = source
            arguments = [
                "prepare_terminal_archive.py",
                "--source-repo", str(repo), "--source-ref", commit,
                "--route-id", "route", "--closeout", closeout,
                "--contract", card, "--conclusion", conclusion,
                "--receipt", "1" * 64, "--destination-repo", str(destination),
                "--commit-and-push",
            ]

            def fetch(receipt, evidence_dir):
                self.assertEqual("1" * 64, receipt)
                return self.receipt_copy(source, evidence_dir)

            with mock.patch.object(ARCHIVE, "fetch_receipt_evidence", side_effect=fetch):
                with mock.patch.object(sys, "argv", arguments):
                    ARCHIVE.main()
            self.assertEqual("", self.git(destination, "status", "--porcelain"))


if __name__ == "__main__":
    unittest.main()
