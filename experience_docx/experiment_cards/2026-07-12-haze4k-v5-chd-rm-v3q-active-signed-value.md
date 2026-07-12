# Haze4K v5 CHD-RM v3q Active Signed-Value Observability

Date: 2026-07-12

Status: `A0B_FORMAL_PASS_A1_SIGNED_LINEAR_PROBE_ONLY`

## Route Identity

- Route type: new representation/policy observability audit. It is not a
  continuation, tolerance repair, or policy replay of v3m, v3n, v3o, or v3p.
- Scientific question: conditional on a nonzero deployable `.125 -> .25`
  candidate change, is there inference-time information that can identify its
  canonical signed first-step value without relying on the easy zero-energy
  abstain class?
- Current GitHub process-rule commit: `af4176c6a66fd8fea9325ddf9ad4a5eeea2e9cd9`.
- Historical evidence source: GitHub `main` paths
  `experience_docx/experiment_cards/2026-07-12-haze4k-v5-chd-rm-v3p-canonical-signed-gain.md`
  and
  `experience_docx/experiment_logs/haze4k_v5_chd_rm_v3p_canonical_signed_gain_20260712/`.
- Source anchor: `github/codex/haze4k-official-arch-anchor` at
  `3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`.
- Route branch: `codex/haze4k-v5-v3q-active-signed-value-20260712`.
- Local WSL route workspace:
  `/home/ubuntu/workspace/ConvIR-B-v3q-active-signed-value-20260712`.
- Cloud `REMOTE_REPO`:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3q-active-signed-value-20260712`.
- Cloud `RUN_ROOT`:
  `/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3q_active_signed_value_20260712`.
- Cloud `EVID_STAGE`:
  `$REMOTE_REPO/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3q_active_signed_value_20260712`.
- Cloud Python:
  `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.

The official ConvIR-B architecture stays unchanged in A0a. Any later sidecar
is an additive route-specific structure and must obey
`OFFICIAL_ARCH_ANCHOR_POLICY.md` and `Haze4K_ARCH_FINETUNE_WORKFLOW.md` before
its first cloud run.

## Evidence And Rationale

v3p established canonical float64 `.125 -> .25` block gains on 1,200
train-derived clean-reference-grouped OOF images for each frozen operator.
v3m established that `direct_step_energy` ranks actionability when zero-action
blocks are included, but its frozen policy replay caused stable image-level
tail harm. The v3q pre-route cloud audit found that the apparent energy AUC is
driven by zero-energy abstentions: on canonical active blocks, energy
beneficial-versus-harmful AUC is about `0.582-0.583`, while its correlation with
`abs(G1)` is about `0.61` and with signed `G1` only about `0.28`.

The single changed variable for v3q is the target/analysis contract: all
scientific observability claims condition on the deterministic active mask
`direct_step_energy > 0`. This does not change the frozen actuator, candidate
ladder, executor, labels, folds, or data.

## Scope And Forbidden Flows

This A0a stage is read-only against v3p canonical cloud outputs. It performs no
candidate inference, selector fitting, feature learning, threshold search,
policy replay, correction retraining, backbone change, canary access, or locked
test access.

Forbidden for the entire route unless a later typed closeout explicitly names
the stage:

- treating zero-energy abstentions as negative examples in an active-block
  signed-value scientific gate;
- relabeling v3p numerical gray-zone abstentions as beneficial or harmful;
- changing the `.125 -> .25` action pair, block16 partition, hard non-overlap
  executor, clean-reference grouped OOF folds, or the frozen D_ref/D_rep
  operators;
- using filename, numeric image id, fold id, or clean-reference id as a model
  feature;
- using a block-level AUC, correlation, or precision result to authorize an
  image-level policy;
- tuning a threshold, coverage, feature family, or model after formal held-out
  evidence;
- route-confirm, canary, or locked Haze4K test access.

## Frozen Label Contract

The label source is the v3p cloud-only canonical table:

```text
/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3p_canonical_signed_gain_20260712/a0_formal/v3p_a0_block_candidate_losses_cloud_only.csv
```

The expected SHA-256 is
`52e6cd8829d37750cfb1e9e2fec39e6ac5cead2e324dbc353df93e5263e89765`, as
recorded by v3p A2. The canonical image reference is:

```text
/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3p_canonical_signed_gain_20260712/a0_formal/v3p_a0_image_candidate_replay_cloud_only.csv
```

For a block `b`, the frozen target is:

```text
G1_b = SSE_b(y_0.125, J) - SSE_b(y_0.25, J)
epsilon_b = epsilon_b(0.125) + epsilon_b(0.25)
```

`G1_b > epsilon_b` is beneficial, `G1_b < -epsilon_b` is harmful, and the
remaining numerical gray zone is abstain. The deployment eligibility mask is
strictly `direct_step_energy > 0`; no learned eligibility rule exists in A0a.

Expected formal counts per operator, inherited from the canonical v3p audit:

| Quantity | Expected value |
| --- | ---: |
| OOF images | 1,200 |
| block rows | 1,088,675 |
| total candidate rows across D_ref/D_rep | 2,177,350 |
| zero-energy abstain rows | 584,680 |
| active rows | 503,995 |
| beneficial active rows | 293,415 (D_ref) / 293,232 (D_rep) |
| harmful active rows | 210,558 (D_ref) / 210,755 (D_rep) |
| active numerical-gray abstain rows | 22 (D_ref) / 8 (D_rep) |

The analysis unit for all uncertainty and later selection claims is the
clean-reference image group. D_ref and D_rep are paired repeated operator
measurements, never independent images.

## A0a Static And Runtime Contract

- Dataset and split: the frozen v3p train-derived 1,200-image OOF
  clean-reference grouped split only; no validation, canary, or locked test.
- Baseline: v3p canonical candidate loss and image replay tables. This is a
  representation audit, not a model-quality comparison.
- Preprocessing and metric: exactly the v3p float64 SSE and numerical gray-zone
  contract. A0a does not recompute or relax it.
- Model/checkpoint loading: none in A0a. Official checkpoint partial-load,
  initialization, and trainable-scope contracts are not applicable until a
  separately authorized candidate-pair feature/model stage.
- Execution profile: `audit/evaluation`, with a specialized table-integrity
  smoke followed by the formal read-only audit.
- Cost: CPU CSV analysis only; no CUDA allocation is allowed or needed.
- Durable runner:
  `experience_docx/tools/run_v3q_a0a_active_signed_contract.sh`.
- A0a audit entrypoint:
  `experience_docx/tools/chd_rm_v3q_a0a_active_signed_contract.py`.

## A0a Controls And Measurements

The runner must write only compact summaries. The raw canonical table remains
at the existing v3p `RUN_ROOT`; it is not copied into this route or GitHub.

| Measurement | Purpose |
| --- | --- |
| SHA-256, row identity, operator/fold/image/block pairing | prevent source drift and duplicate/missing rows |
| zero-energy, active numerical-gray, beneficial, harmful counts | freeze the conditional estimand without relabeling the gray zone |
| energy AUC: beneficial vs all non-beneficial and vs harmful-only | expose abstain-driven metric inflation |
| energy correlation with G1 and abs(G1) | distinguish direction from magnitude information |
| active-block energy saturation and harmful rate at the maximum | prohibit threshold-only continuation |
| per-image active G1 aggregation and D_ref/D_rep agreement | establish the later image-level calibration unit |
| filename/fold/clean-id feature audit | forbid metadata leakage before any feature table exists |

## Gates

| Stage | Question | Gate type and independent contract | `PASS` authorizes |
| --- | --- | --- | --- |
| A0a smoke | Does the runner read only the pinned canonical source and reproduce the 32-image prefix row/operator structure without writing raw artifacts? | structural `PASS/FAIL`; pinned source path, SHA-256, row identities, zero CUDA allocation, no locked data | A0a formal only |
| A0a formal | Does the full pinned table reproduce the frozen active-stratum counts, paired grouping, and magnitude-versus-sign diagnostic without leakage? | structural/numerical `PASS/FAIL`; exact source hash, expected counts above, no duplicate or missing `(operator,name,block_y,block_x)` key, no forbidden fields selected | a separately contracted A0b candidate-pair feature-table stage only |
| A0a formal | Does the audit show that active-only signed discrimination remains an open scientific question rather than an energy-threshold problem? | diagnostic only; no numeric result can authorize a policy or model | no additional authorization |

Any source hash, count, pairing, forbidden-data, or metadata-feature violation is
`FAIL` and stops v3q under this label source. An engineering read failure is
`INCONCLUSIVE` and authorizes only the same-stage transport/read repair. A0a
never authorizes scorer training, policy replay, canary, or locked-test access.

## Prospective A0b Boundary

Only after A0a PASS may a new stage define a compact candidate-pair feature
table. Its feature contract must contain inference-time tensors/statistics only:
hazy input, `y_0.125`, `y_0.25`, midpoint, candidate delta, clipping/distance
features, and explicitly frozen context features. It must include energy-only,
unsigned-magnitude-only, within-image G1 shuffle, and metadata-only controls;
use per-image weighting; and reserve nested inner calibration before any policy
replay. A0a does not pre-authorize a sidecar architecture, loss, threshold,
coverage, or policy.

## Evidence Boundary And Decision

- Raw artifacts retained on cloud: source table, formal log, and any temporary
  feature table.
- Compact route artifacts: A0a source manifest, row/count summary, control
  summary, typed closeout, status excerpt, and evidence README.
- A0a decision label: pending.
- `PASS` authorizes: only A0b candidate-pair feature-table contract design.
- `INCONCLUSIVE` authorizes: only A0a read/transport repair.
- `FAIL` stops: v3q use of this canonical label source; no threshold or policy
  continuation.

## Completed Evidence

The A0a 32-image smoke completed on `convir-4090` at
`2026-07-12T20:10:29+08:00` with
`V3Q_A0A_SMOKE_PASS_AUTHORIZE_FORMAL_ONLY`.

- The pinned canonical block SHA-256 matched
  `52e6cd8829d37750cfb1e9e2fec39e6ac5cead2e324dbc353df93e5263e89765`.
- Both operators read exactly 20,000 block rows across the same 32
  clean-reference groups; no metadata was used as a model feature, and no GPU,
  training, canary, or locked-test operation occurred.
- The smoke active-only beneficial-versus-harmful energy AUC was `0.57005`
  (`D_ref`) and `0.57508` (`D_rep`), confirming the previously observed
  magnitude/sign gap without treating the result as a policy gate.
- The typed closeout authorizes A0a formal only. The formal stage must repeat
  dynamic preflight with a fresh `a0a_formal` output path.

Compact evidence is under
`experience_docx/experiment_logs/haze4k_v5_chd_rm_v3q_active_signed_value_20260712/`.

## A0a Formal Result

The formal read-only audit completed on `convir-4090` at
`2026-07-12T20:18:17+08:00` with
`V3Q_A0A_FORMAL_PASS_AUTHORIZE_A0B_FEATURE_CONTRACT_ONLY`.

- The v3p canonical block SHA-256 and all expected counts matched exactly:
  1,088,675 rows and 503,995 active rows per operator, with 22 D_ref and 8
  D_rep active numerical-gray abstentions retained as abstentions.
- Active-only energy correlation is `0.22728` / `0.22959` with signed G1 and
  `0.48707` / `0.49025` with absolute G1. The latter remains materially larger;
  zero-energy rows must not be included in a signed-value claim.
- Active beneficial-versus-harmful energy AUC is `0.58226` / `0.58328`, while
  the legacy all-nonbeneficial AUC is `0.88940` / `0.88959` because of the
  large zero-energy abstain class.
- At the exact maximum energy, 29.53% / 30.13% of active blocks are harmful.
  This closes threshold-only escalation as an A0b continuation.
- D_ref/D_rep agreement is stable at the image-group unit: sum-G1 Pearson
  `0.99830`, active-energy Pearson `0.99944`, and within-image energy/G1
  correlation Pearson `0.95960`.

This is a structural/diagnostic result, not evidence for a learned scorer or a
deployed policy. Only a separately specified A0b candidate-pair feature
contract is authorized.

## A0b Candidate-Pair Feature Contract

A0b asks a narrower engineering question: can v3q regenerate the frozen v3p
candidate pair and attach a strictly inference-time, active-block scalar feature
schema while preserving the canonical key and signed-label contract?

The frozen candidate producer is source code only, not governance authority:

```text
V3P_REMOTE_REPO=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3p-canonical-signed-gain-20260712
V3P_SOURCE_COMMIT=555fd008e29f02128564f2fad41d0095ee44f5ea
```

It must use the exact v3p assets and SHA-256 contract already recorded by A0a:
official A0 checkpoint, frozen control checkpoint, v3j split/bounds, v3l
operator artifacts, density artifact, D7c artifact, and fixed OOF reference.
Candidate construction remains block16, hard non-overlap, clamp, D_ref/D_rep,
and action pair `.125 -> .25` exactly as in v3p. The v3p canonical block CSV is
the label source and later identity reference; it is never copied into v3q.

The raw A0b cloud feature table may contain metadata solely for grouping and
identity, plus separate target columns. The following are forbidden model
features: `name`, `index`, `fold`, `clean_reference_group`, numeric image id,
operator label, GT pixels, canonical candidate SSE, G1, G1 state, and any
target-derived aggregate.

The frozen inference-time scalar schema is:

```text
direct_step_energy, d7c_score_mean,
delta channel mean/std, midpoint channel mean,
<I - midpoint, delta>,
<grad(I - midpoint), grad(delta)>,
per-channel covariance(I - midpoint, delta),
hazy luminance mean/std, hazy saturation mean,
raw-candidate clip fractions and signed distance-to-clip at .125/.25
```

`I`, `y_.125`, `y_.25`, midpoint, delta, and raw pre-clamp candidates are all
computed from the frozen inference path before labels are consulted. A0b does
not export FAM feature maps, context patches, image crops, or learned features.
Those require a later contract if the linear probe is insufficient.

| Stage | Question | Gate | `PASS` authorizes |
| --- | --- | --- | --- |
| A0b smoke | Do 32 paired images reproduce active-row keys, canonical G1 states, direct-step energy, and no-leak schema under the pinned producer? | structural `PASS/FAIL`; exact 32-image name/group pair, active-row count, no duplicate key, no forbidden model field, canonical state equality, direct-energy and G1 numerical match within the source gray-zone contract | A0b formal only |
| A0b formal | Does the 1,200-image table reproduce all A0a active counts and paired keys while retaining only inference-time model fields? | structural `PASS/FAIL`; v3p source/assets hashes, 5-fold groups, active counts, row identity, schema allowlist, no target or metadata in feature list | A1 signed-linear probe only |

The raw feature table stays in `RUN_ROOT`. A0b copies only schema, source
manifest, compact operator/fold counts, typed closeout, and README text into
`EVID_STAGE`. No A0b result authorizes sidecar training, threshold calibration,
policy replay, canary, or locked-test access.

## A0b Smoke Result

The 32-image candidate-pair feature-contract smoke completed on
`convir-4090` at `2026-07-12T20:45:30+08:00` with
`V3Q_A0B_SMOKE_PASS_AUTHORIZE_FORMAL_ONLY`.

- The pinned v3p source commit, canonical source SHA-256, frozen assets, and
  route commit all matched their recorded contracts.
- Each operator emitted and canonically verified `14,151` active rows. The
  D_ref counts were `9,177` beneficial and `4,974` harmful; D_rep had `9,173`
  beneficial and `4,978` harmful. Both active gray counts were zero.
- Exact ordered key checks passed, with maximum direct-energy and G1
  differences both `0.0` against the canonical source.
- The 24 model feature columns were disjoint from metadata, target, and
  forbidden-field lists. The raw feature CSV remains under cloud `RUN_ROOT`.
- This stage performed no training, canary, or locked-test operation.

Only `v3q-A0b-formal` is now authorized. It must reproduce the complete
1,200-image feature contract before a signed linear probe can be considered.

## A0b Formal Result

The 1,200-image-per-operator candidate-pair feature contract completed on
`convir-4090` at `2026-07-12T21:32:07+08:00` with
`V3Q_A0B_FORMAL_PASS_AUTHORIZE_A1_SIGNED_LINEAR_PROBE_ONLY`.

- Each operator emitted exactly `503,995` active rows. D_ref retained `22`
  numerical-gray abstentions (`293,415` beneficial, `210,558` harmful); D_rep
  retained `8` (`293,232` beneficial, `210,755` harmful).
- The cloud-only table contains `1,007,990` active rows plus its header. All
  canonical key, G1-state, direct-energy, and signed-G1 checks passed with
  maximum observed energy/G1 difference `0.0`.
- The 24 inference-time feature columns remained disjoint from metadata,
  labels, and the forbidden-field list. No training, canary, or locked-test
  action occurred.

Only a signed linear probe is authorized next. It must use clean-reference
grouped outer folds, per-image rather than per-block weighting, and report the
energy-only, unsigned-magnitude-only, within-image shuffled-label, and
metadata-only negative controls. It may not tune or replay a policy, choose a
deployment threshold, create a sidecar, or access canary/locked data.
