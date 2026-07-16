# <Route Name>

Date: <YYYY-MM-DD>

Status: <DRAFT | PLANNED | RUNNING | STOPPED | COMPLETED>

## Identity

- Route id:
- Question:
- GitHub rules commit and canonical rule-bundle digest:
- Source branch/commit:
- Route branch:
- Local editing workspace:
- Cloud workspace policy: <fresh_route | exact_continuation>
- Cloud run root:
- Explicit cloud Python:
- Locked test/canary policy:

## Scientific Contract

- Target population and analysis/grouping unit:
- Intervention or factor contrast:
- Reference:
- Primary outcome, direction, and aggregation:
- Claim type: <causal | associational | predictive>
- Preferred mechanism:
- Null and strongest competing explanation:
- Cheapest observation that separates them:
- Minimum worthwhile effect or risk limit and independent source:
- Primary gate and uncertainty estimator:
- `PASS` authorizes:
- `INCONCLUSIVE` authorizes:
- `FAIL` stops:

## Design And Evidence Roles

- Design: <paired_ablation | factorial | feasibility_oracle | screening_confirmation | cross_fit_confirmation | adaptive>
- Experimental assignment/pairing/blocking:
- Sample/group/fold/seed count and justification:
- Multiplicity treatment:
- Missing/exclusion policy:
- Evidence-role ledger:

| Evidence | Role | Allowed use | Forbidden use |
| --- | --- | --- | --- |
| | <engineering_debug | development_screening | confirmation | sealed_final> | | |

- Candidate/operator/threshold freeze point:
- Forbidden continuations/evidence reuse:

## Implementation Contract

- Exact change and enabled mechanism:
- Explicitly disabled mechanisms:
- Checkpoint/load/init/freeze contract:
- Input whitelist and prohibited inputs:
- No-op/neutral behavior:
- Dataset/split/preprocessing/metric identities:
- Matched baseline:
- Parameter/MAC hard limit, if decision-relevant:
- Latency/memory hard limit or descriptive-only rationale:
- Required asset manifest:

## Stages

Keep only stages that can change a written next action.

| Stage | Evidence role and scope | Gate | Pass authorizes |
| --- | --- | --- | --- |
| engineering smoke | | | |
| formal confirmation | | | R3 terminal review |

- First authorized stage:
- Integrated smoke checks:
- Expected phase/wall-time budget:
- Heartbeat and monitor profile:
- Maximum observation windows and escalation condition:
- Unit-boundary resume policy:

## Outputs And Closeout

- Runner:
- Operations manifest:
- Status/log/closeout paths:
- Required retained states and hashes:
- Compact GitHub evidence:
- Cloud-only raw artifacts:
- Terminal archive updates:

## Decision

Fill only after terminal evidence is available.

- Verdict:
- Primary metric/gate reason:
- Mechanism/control reason:
- Preservation/safety reason:
- Evidence independence reason:
- Cost/deployability reason:
- Authorized next action or terminal stop:
