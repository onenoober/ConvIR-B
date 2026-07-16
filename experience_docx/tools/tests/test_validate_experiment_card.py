"""Generic compact route-card validator tests."""

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
- GitHub rules commit and canonical rule-bundle digest: commit and digest
- Source branch/commit: source
- Route branch: codex/route
- Local editing workspace: /workspace/route
- Cloud workspace policy: fresh_route
- Cloud run root: /runs/route
- Explicit cloud Python: /env/python
- Locked test/canary policy: blocked

## Scientific Contract

- Target population and analysis/grouping unit: images
- Intervention or factor contrast: candidate versus anchor
- Reference: anchor
- Primary outcome, direction, and aggregation: higher paired score
- Claim type: predictive
- Preferred mechanism: readable signal
- Null and strongest competing explanation: no signal
- Cheapest observation that separates them: paired control
- Minimum worthwhile effect or risk limit and independent source: fixed prior threshold
- Primary gate and uncertainty estimator: paired image bootstrap
- `PASS` authorizes: review
- `INCONCLUSIVE` authorizes: review only
- `FAIL` stops: route

## Design And Evidence Roles

- Design: cross_fit_confirmation
- Experimental assignment/pairing/blocking: paired images
- Sample/group/fold/seed count and justification: four folds
- Multiplicity treatment: one primary family
- Missing/exclusion policy: none
- Candidate/operator/threshold freeze point: before confirmation
- Forbidden continuations/evidence reuse: no tuning

| Evidence | Role | Allowed use | Forbidden use |
| --- | --- | --- | --- |
| smoke | engineering_debug | integrity | result |

## Implementation Contract

- Exact change and enabled mechanism: one frozen head
- Explicitly disabled mechanisms: policy and locked test
- Checkpoint/load/init/freeze contract: exact
- Input whitelist and prohibited inputs: deployable only
- No-op/neutral behavior: exact zero
- Dataset/split/preprocessing/metric identities: fixed
- Matched baseline: anchor
- Parameter/MAC hard limit, if decision-relevant: fixed
- Latency/memory hard limit or descriptive-only rationale: descriptive
- Required asset manifest: assets.json

## Stages

| Stage | Evidence role and scope | Gate | Pass authorizes |
| --- | --- | --- | --- |
| S0 | engineering_debug | integrity | formal |

- First authorized stage: S0
- Integrated smoke checks: identity and finite update
- Expected phase/wall-time budget: 30 minutes
- Heartbeat and monitor profile: 60 seconds, short
- Maximum observation windows and escalation condition: three; stale
- Unit-boundary resume policy: complete fold only

## Outputs And Closeout

- Runner: experience_docx/tools/run_route.sh
- Operations manifest: experience_docx/route_operations.json
- Status/log/closeout paths: run root
- Required retained states and hashes: final folds
- Compact GitHub evidence: README and closeout
- Cloud-only raw artifacts: predictions and states
- Terminal archive updates: index and card
"""


class CardValidatorTests(unittest.TestCase):
    def check(self, text=CARD, launch_ready=True):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "card.md"
            path.write_text(text, encoding="utf-8")
            return VALIDATOR.validate(path, launch_ready)

    def test_compact_card_passes(self):
        errors, digest = self.check()
        self.assertEqual([], errors)
        self.assertEqual(64, len(digest))

    def test_placeholder_fails(self):
        errors, _ = self.check(CARD.replace("route\n", "<route>\n", 1))
        self.assertTrue(errors)

    def test_dispatcher_section_fails(self):
        errors, _ = self.check(CARD + "\n## Agent Execution Routing\n\n- dispatcher: enabled\n")
        self.assertTrue(any("dispatcher" in error for error in errors))

    def test_launch_ready_requires_planned(self):
        errors, _ = self.check(CARD.replace("Status: PLANNED", "Status: DRAFT"))
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
