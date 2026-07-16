"""Tests for the minimal generic route-card validator."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "validate_experiment_card.py"
SPEC = importlib.util.spec_from_file_location("validate_experiment_card", MODULE_PATH)
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)

CARD = """# Route

Date: 2026-07-16

Status: PLANNED

## Identity

- Route id: route
- Question: Does the frozen intervention help?
- Rules commit: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
- Source branch/commit: source
- Route branch: codex/route
- Locked test/canary policy: blocked

## Scientific Contract

- Population and analysis/grouping unit: images
- Intervention or factor contrast and reference: candidate versus anchor
- Primary outcome, direction and aggregation: higher paired score
- Preferred mechanism and strongest competing explanation: signal versus proxy
- Evidence roles and candidate/freeze point: engineering_debug then confirmation; freeze before confirmation
- Primary gate, uncertainty and threshold source: paired bootstrap; fixed prior threshold
- `PASS` authorizes: formal review
- `INCONCLUSIVE` authorizes: written additional evidence only
- `FAIL` stops: route

## Implementation Contract

- Exact change and disabled mechanisms: one head; policy disabled
- Checkpoint/load/init/freeze contract: exact
- Input whitelist and prohibited inputs: deployable only
- Dataset/split/preprocessing/metric identities: fixed
- Matched baseline and budget: anchor at matched budget
- Resource/cost limits or descriptive-only rationale: descriptive latency
- Runner and required assets: experience_docx/tools/run_route.sh; assets.json

## Operations And Evidence

| Operation | Evidence role/scope | Gate | Pass authorizes |
| --- | --- | --- | --- |
| S0 | engineering_debug | integrity | formal |

- First operation: S0
- Expected wall time and monitor profile: 30 minutes; short
- Complete-unit resume policy: complete fold only
- Cloud workspace/run/output/status/closeout: fresh paths
- Compact Git evidence and cloud-only raw artifacts: README/closeout; raw predictions on cloud
"""


class CardValidatorTests(unittest.TestCase):
    def check(self, text=CARD, launch_ready=True):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "card.md"
            path.write_text(text, encoding="utf-8")
            return VALIDATOR.validate(path, launch_ready)

    def test_minimal_card_passes(self):
        errors, digest = self.check()
        self.assertEqual([], errors)
        self.assertEqual(64, len(digest))

    def test_placeholder_fails(self):
        errors, _ = self.check(CARD.replace("route\n", "<route>\n", 1))
        self.assertTrue(errors)

    def test_retired_dispatcher_pattern_fails(self):
        errors, _ = self.check(CARD + "\n- dispatcher: enabled\n")
        self.assertTrue(any("retired control pattern" in error for error in errors))

    def test_launch_ready_requires_planned(self):
        errors, _ = self.check(CARD.replace("Status: PLANNED", "Status: DRAFT"))
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
