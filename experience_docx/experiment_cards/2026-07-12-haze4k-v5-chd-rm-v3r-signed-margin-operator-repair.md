# Haze4K v5 CHD-RM v3r First-Step Signed-Margin Operator Repair

Date: 2026-07-12

Status: `PLANNED_A0_PRIVILEGED_REPAIR_GEOMETRY_ONLY`

## Route Identity

- Route type: new bounded operator-repair geometry audit. It is not a continuation
  of v3q signed scoring, a threshold/calibration repair, or a policy replay.
- Scientific question: on the frozen v3p first-step operators, is there enough
  anchor-preserving, locally repairable signed-margin ceiling to justify changing
  the correction operator rather than adding another post-hoc scorer?
- Current GitHub process-rule commit: `f999591b8d6d09beef3079d51db6eac53d1ae302`.
- Historical evidence source: `experience_docx/CHD_RM_EXPERIMENT_INDEX.md`, the
  v3p route card/evidence, and the v3q route card/evidence on GitHub `main`.
- Source anchor: `github/codex/haze4k-official-arch-anchor` at
  `3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`.
- Route branch: `codex/haze4k-v5-v3r-signed-margin-operator-repair-20260712`.
- Local WSL route workspace:
  `/home/ubuntu/workspace/ConvIR-B-v3r-signed-margin-operator-repair-20260712`.
- Cloud `REMOTE_REPO`:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3r-signed-margin-operator-repair-20260712`.
- Cloud `RUN_ROOT`:
  `/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3r_signed_margin_operator_repair_20260712`.
- Cloud `EVID_STAGE`:
  `$REMOTE_REPO/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3r_signed_margin_operator_repair_20260712`.
- Cloud Python:
  `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.

## Evidence And Boundary

v3p established that a constrained oracle can beat uniform `.25` by about
`.0213-.0216 dB` LCB95 without selected harmful SSE. v3q then closed learned
signed scoring from its 24 scalar schema: full weighted AUC only exceeded the
unsigned control by `.00111/.00075`, while within-image shuffled labels retained
most apparent discrimination. v3r therefore changes the question from whether
the old correction can be scored after the fact to whether it can be repaired
at its first signed step.

The frozen v3p source is:

```text
/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3p_canonical_signed_gain_20260712/a0_formal/v3p_a0_block_candidate_losses_cloud_only.csv
```

Its required SHA-256 is
`52e6cd8829d37750cfb1e9e2fec39e6ac5cead2e324dbc353df93e5263e89765`.
The historical candidate producer source is pinned at cloud repository commit
`555fd008e29f02128564f2fad41d0095ee44f5ea`; it supplies code and frozen assets,
not current process rules.

## Scope And Forbidden Flows

The sole authorized stage is A0, a privileged train-derived 1,200-image grouped
OOF audit. It reconstructs the old operators exactly, uses clean targets only to
measure repair ceilings, and performs no learning.

Forbidden for this route until a typed later closeout explicitly authorizes it:

- training a repair, value, confidence, selector, calibration, or sidecar head;
- threshold or coverage search, policy replay, canary access, or locked-test access;
- reopening the v3p scalar-A physics package;
- changing the `.125/.25` action pair, block16 partition, hard non-overlap
  executor, OOF groups, frozen D_ref/D_rep artifacts, or baseline checkpoint;
- treating any privileged A0 repair as a deployable policy;
- backbone unfreezing or changing the official ConvIR-B architecture.

## A0 Repair Geometry

For each active block, let `y0` be the A0 prediction, `r` the frozen output
step, and `L_alpha(r) = SSE(clip(y0 + alpha * r), J)`. The signed first-step
gain is `G1(r) = L_.125(r) - L_.25(r)`. Every candidate must preserve the old
anchor, `L_.125(r') <= L_.125(r) + epsilon`, and satisfy
`G1(r') > epsilon`, where `epsilon` is the pinned v3p numerical SSE tolerance
for the two candidate losses. Actual clamp is always included.

The fixed grid has 65 points `0, 1/64, ..., 1`; results are fixed-grid
feasible ceilings, not claims of a continuous global optimum.

| Repair | Privileged candidate family | Interpretation |
| --- | --- | --- |
| scale | `r' = lambda r`, `lambda in [0,1]` | Tests whether abstention or overshoot alone explains the gap. |
| channel scale | `r'_c = lambda_c r_c` | Tests whether RGB-channel amplitude/color mismatch explains the gap. |
| direction line | `r' = (1-gamma)r + gamma r_GT`, `r_GT = gate * clip(4(J-y0), -B, B)` | A bounded low-capacity direction repair that preserves the old gated support. |
| direct-clean | `r_GT` | Privileged upper reference only, never a train target or policy. |

For scale and direction-line repairs, A0 records both the candidate maximizing
`G1` and the smallest feasible grid displacement. Channel scales maximize the
separable RGB contribution subject to each channel's anchor constraint. A block
without a feasible positive-margin candidate remains explicitly unrepaired;
it is never silently relabelled as safe.

Old harmful blocks are assigned deterministically to `wrong_direction` when
their unclipped `sum((J-y0)*r)` is non-positive, otherwise to
`harmful_overshoot`. An old beneficial block with harmful `.25 -> .5` gain is
`beneficial_but_oversized`; all remaining blocks are `conservative_or_ok`.

## A0 Measurements

For each operator and repair type, A0 writes cloud-only block/image rows and
compact summaries of:

- repairable active-block fraction and class-stratified fraction;
- median/p90 angular change and relative residual-norm change;
- bound saturation and `.25` clip fraction before/after;
- repaired `.25` PSNR lift versus old uniform `.125` and old uniform `.25`;
- signed harmful/beneficial SSE ratio, p10, p05, CVaR5, and severe count at the
  image group unit;
- D_ref/D_rep paired agreement and a worst-operator decision.

The repair outcomes are generated through the same hard block executor used by
the old operator. v3p already established additive versus hard-mosaic agreement;
this route does not optimize an executor.

## Gates

| Stage | Gate | Decision and authorization |
| --- | --- | --- |
| A0 smoke | Structural: source SHA, 32 frozen sorted OOF names, paired row keys/counts, old gain/state reconstruction within source epsilon, no forbidden split, no raw artifact copied into evidence. | `PASS` authorizes A0 formal only. Any mismatch is `FAIL`; transport/read issues are `INCONCLUSIVE` for same-stage repair only. |
| A0 formal | Structural: full 1,200 images/operator, exact source SHA/keys/counts, old gain/state agreement, same frozen assets, and compact/raw boundary. | Structural mismatch is `FAIL`. |
| A0 formal | Scientific ceiling: a repair family has paired clean-reference-group bootstrap LCB95 lift over old uniform `.25` of at least `+0.005 dB` on both operators. | Scale pass authorizes only a scale-repair route-design stage. Channel/direction pass with scale failure authorizes only the corresponding repair representation/training-contract design stage. |
| A0 formal | Safety diagnosis: report severe/hard counts and tail metrics relative to old `.125`; this privileged audit cannot authorize deployment. | Any tail result is diagnostic until a future frozen learned operator has direct replay evidence. |

`+0.005 dB` is the predeclared SESOI: approximately 21% of v3p's constrained
oracle gap over uniform `.25`, not a threshold selected from v3r results.

If scale, channel-scale, and direction-line ceilings all fail the dual-operator
SESOI, decision is `V3R_A0_REPAIR_CEILING_FAIL_REDESIGN_ACTION_PARAMETERIZATION`.
If scale fails but direction-line passes, the intended conclusion is direction
repair is required; it does not authorize a frozen post-hoc scorer.

## Static Runtime Contract

- Dataset/split: the v3p train-derived, clean-reference-grouped 1,200-image OOF
  source only; smoke is its first 32 frozen names.
- Operators: exactly `D_ref` and `D_rep`, with their v3l artifacts, D7c gate,
  density artifact, A0 checkpoint, control checkpoint, bounds, and reference
  rows pinned to the v3p source contracts.
- Metric: the frozen v3p float32 add-and-clamp candidate renderer, followed by
  float64 RGB SSE reduction with v3p numerical tolerance; image PSNR derives
  from that rendered full-image MSE. A `float64` render is forbidden because it
  is not replay-equivalent to the frozen v3p candidate path.
- Analysis unit: clean-reference image group. D_ref/D_rep are paired repeated
  measurements, never independent samples.
- Profile: `audit/evaluation`: cloud smoke then formal audit.
- Durable runner: `experience_docx/tools/run_v3r_a0_privileged_repair_geometry.sh`.
- Audit entrypoint: `experience_docx/tools/chd_rm_v3r_a0_privileged_repair_geometry.py`.

## Evidence Boundary

Cloud-only: model outputs, per-block candidate/repair rows, per-image rows,
logs, and temporary tensors. Compact evidence: source manifest, closeout,
repair-type and category summaries, dual-operator decision, runner status
excerpt, and evidence README. A0 updates GitHub `main` only at its terminal
decision or explicit major handoff under the current sync protocol.

Initial decision label:

```text
V3R_START_PRIVILEGED_OPERATOR_REPAIR_GEOMETRY_ONLY
PARENT_F999591B
V3Q_HISTORICAL_FAIL_UNCHANGED
NO_SCORER_TRAINING
NO_POLICY_REPLAY
NO_THRESHOLD_SEARCH
NO_BACKBONE_UNFREEZE
NO_PHYSICS_REOPEN
NO_CANARY
NO_LOCKED_TEST
```
