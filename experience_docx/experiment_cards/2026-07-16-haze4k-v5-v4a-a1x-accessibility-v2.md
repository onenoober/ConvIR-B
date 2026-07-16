# Haze4K v5 CHD-RM v4a-A1X Accessibility v2

Date: 2026-07-16

Status: `PLANNED`

## Identity

- Route id: `haze4k_v5_chd_rm_v4a_a1x_accessibility_v2_20260716`.
- Question: Can one frozen global nonlinear head recover material, image-conditioned safe direction from inference-eligible tensors on the adequate exact-half interface?
- GitHub rules commit and canonical rule-bundle digest: `713aa5e2aa5a9c0dc1e09738dfb129db7327d9fd`; `629a1125edba1bbc376aae3a90252d50f64c7769590731fa779d5ffe841053ae`.
- Source branch/commit: immutable `github/codex/haze4k-official-arch-anchor@3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`.
- Route branch: `codex/haze4k-v5-v4a-a1x-accessibility-v2-20260716`.
- Local editing workspace: `/home/ubuntu/workspace/ConvIR-B-v4a-a1x-accessibility-v2-20260716`.
- Cloud workspace policy: `fresh_route` for S0 and formal; `exact_continuation` only for the preregistered formal resume operation.
- Cloud run root: `/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v4a_a1x_accessibility_v2_20260716`.
- Explicit cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Locked test/canary policy: blocked for every stage; neither may be listed, read, evaluated, selected, or used for repair.

## Scientific Contract

- Target population and analysis/grouping unit: the untouched 432 names at frozen indices `768:1200`; clean-reference image is the paired bootstrap and grouping unit.
- Intervention or factor contrast: true target versus operator-and-native-shape-blocked cyclic target shuffle under one identical fixed head, seed, fold, optimizer, and budget.
- Reference: privileged shrink for utility, exact-half privileged direction for attainable retention, shuffled mapping for image conditioning, and old `.125`/`.25` for pointwise safety.
- Primary outcome, direction, and aggregation: higher safe rendered PSNR; per-image cells/operators stay paired and the worse `D_ref`/`D_rep` operator is selected inside every image-bootstrap draw.
- Claim type: predictive accessibility under this exact head and data contract; not a candidate, calibrated executor, action policy, canary, promotion, or locked-test claim.
- Preferred mechanism: multiscale nonlinear aggregation plus global context makes the adequate exact-half direction materially accessible from the five allowed deployable tensors.
- Null and strongest competing explanation: the signal remains submaterial, or the head learns only a population prior that the matched shuffled-target control reproduces.
- Cheapest observation that separates them: complete four-fold outer OOF predictions for only the true and shuffled cells on all 432 untouched names.
- Minimum worthwhile effect or risk limit and independent source: safe-gain LCB95 `>=+0.020 dB`, exact-half retention LCB95 `>=0.25`, and true-minus-shuffle LCB95 `>=+0.005 dB`, fixed before A1X access from the A1R/A1C contracts on `github/main`.
- Primary gate and uncertainty estimator: intersection of the three primary LCB95 thresholds under 4,000 paired image bootstraps; exact-half adequacy and pointwise safety are eligibility guards, while repairable fraction is supporting only.
- `PASS` authorizes: `R3_REVIEW_ONLY` for a separate direct-action design; it does not authorize a candidate, policy, canary, or locked test.
- `INCONCLUSIVE` authorizes: `R3_REVIEW_ONLY`; no tuning, result-conditioned rerun, data expansion, confirmation reuse, or promotion.
- `FAIL` stops: this exact `A1X_ACCESS_*` head/input/training contract; no same-data repair or alternate head search.

## Design And Evidence Roles

- Design: `cross_fit_confirmation` with four deterministic outer folds of 108 names and two paired cells.
- Experimental assignment/pairing/blocking: fold is local index modulo four; target shuffle is cyclic plus one only within outer-train operator/native-shape blocks; true/shuffle and both operators remain paired by image.
- Sample/group/fold/seed count and justification: all 432 untouched names, four complete folds, seed 3407, eight total fold/cell units; no post-result expansion or second seed claim.
- Multiplicity treatment: one preregistered algorithm and one matched negative control; the three primary measures form an intersection-union gate, so every LCB threshold must pass.
- Missing/exclusion policy: no exclusion; any missing name, fold, cell, operator, finite value, state hash, or complete OOF row fails integrity before scientific interpretation.
- Evidence-role ledger:

| Evidence | Role | Allowed use | Forbidden use |
| --- | --- | --- | --- |
| A1R indices `256:288` | `engineering_debug` | S0 shapes, no-op, finite gradient, parameter/MAC checks | thresholds, head choice, or scientific claim |
| A1F/A1R/A1C compact evidence | `development_screening` | preregister interface, target, safety, and thresholds | A1X fitting or result-conditioned rescue |
| A1X outer-train partition | `development_screening` | fit normalization and its true/shuffle state only | inspect paired heldout outcomes or select a variant |
| A1X outer-heldout partitions | `confirmation` | one terminal complete OOF decision | fitting, early stopping, tuning, repair, or rerun selection |
| canary and Haze4K locked test | `sealed_final` | none | all access and evaluation |

- Candidate/operator/threshold freeze point: this card freezes inputs, head widths, folds, shuffle, optimizer, epochs, seed, operators, renderer, support/bounds, bootstrap, gates, and outcomes before any A1X read.
- Forbidden continuations/evidence reuse: no old A1X workspace/manifest/validator/authorization reuse; no A1X result exposure before eight unit hashes; no tuning or re-selection from confirmation; no external model task; no canary/locked test.

## Implementation Contract

- Exact change and enabled mechanism: standalone `A1X_ACCESS_Head`, 15-channel `1x1 -> 24/48/96` depthwise-separable encoder, global-average context, symmetric decoder, and bounded three-channel exact-half correction.
- Explicitly disabled mechanisms: official ConvIR edits, backbone/current-controller updates, local-head family search, interface change, loss/grid/support/bounds tuning, policy/candidate selection, and sealed-data access.
- Checkpoint/load/init/freeze contract: official/current scientific dependencies use the hashes in `a1x_v2_asset_manifest.json`; all historical parameters stay frozen/eval; only the new head trains; final projection is exactly zero initialized.
- Input whitelist and prohibited inputs: only exact-half hazy RGB, frozen base RGB, old `.125` RGB, old `.25` RGB, and current Delta-u; clean RGB, target Delta-u, safe-grid winner, action amplitude, outer outcomes, canary, and test are prohibited forward inputs.
- No-op/neutral behavior: zero final projection must yield bit-exact current Delta-u before the first update.
- Dataset/split/preprocessing/metric identities: frozen 1,200-name order; A1X `768:1200`; A1R bilinear exact-half transport; A1F target/grid/safe selector; paired `D_ref/D_rep`; float32 add-and-clamp renderer.
- Matched baseline: privileged shrink for utility; exact-half privileged target for adequacy/retention; identical shuffled-target head for the negative control.
- Parameter/MAC hard limit, if decision-relevant: added parameters `<=300,000`; largest exact-half head MACs `<=600,000,000`; exceeding either is S0 engineering fail.
- Latency/memory hard limit or descriptive-only rationale: descriptive only because this route diagnoses accessibility and does not produce a deployable candidate; hard latency/memory gates would not answer the scientific question.
- Required asset manifest: `experience_docx/tools/a1x_v2_asset_manifest.json`; hashes and pinned dependency commits must match before source reconstruction.

## Stages

| Stage | Evidence role and scope | Gate | Pass authorizes |
| --- | --- | --- | --- |
| `A1X_V2_S0` | `engineering_debug`, 32 already-consumed A1R names, no A1X access | asset/source identity, 15-channel whitelist, exact no-op, finite two-cell update, two native shapes, parameter/MAC limits | `A1X_V2_FORMAL_ONLY` |
| `A1X_V2_FORMAL` | `confirmation`, all 432 A1X names, four folds x true/shuffle | complete unit hashes and OOF rows, exact-half adequacy, pointwise safety, three primary paired-bootstrap gates | R3 terminal review only |

- First authorized stage: `A1X_V2_S0`.
- Integrated smoke checks: pinned assets/source, A1X untouched, load/freeze, five-input whitelist, exact no-op, finite forward/backward, nonzero first gradients, two native shapes, parameter/MAC hard limits.
- Expected phase/wall-time budget: S0 at most 30 minutes; formal cache 3 hours, eight fold/cell units 5 hours, terminal replay/summary 2 hours, total 10 hours hard budget.
- Heartbeat and monitor profile: 30-second heartbeat; schema-v3 `short` for S0 and `long` for formal, each finish call observes at most 60 seconds.
- Maximum observation windows and escalation condition: one window per user turn; stale heartbeat, dead session without closeout, identity mismatch, or command failure escalates once and stops polling.
- Unit-boundary resume policy: only complete `fold x cell` state files whose contract hash binds route commit, runner SHA, configuration, train names, and heldout names; partial units are rerun and no intermediate metric is exposed.

## Outputs And Closeout

- Runner: `experience_docx/tools/run_chd_rm_v4a_a1x_accessibility_v2.sh`.
- Operations manifest: `experience_docx/route_operations.json`, schema v3; initial commit contains only `A1X_V2_S0`, later formal manifest requires the exact S0 closeout tuple.
- Status/log/closeout paths: raw output under the cloud run root; compact closeout under `experience_docx/experiment_logs/haze4k_v5_chd_rm_v4a_a1x_accessibility_v2_20260716/`.
- Required retained states and hashes: eight formal fold/cell unit states plus `a1x_v2_unit_manifest.json`; each state and contract is SHA-256 bound.
- Compact GitHub evidence: route card, typed closeout, compact formal/S0 summary, unit manifest without state payloads, and evidence README.
- Cloud-only raw artifacts: runtime log, raw OOF rows, cached tensors, head/optimizer state payloads, and any arrays/images.
- Terminal archive updates: `EXPERIMENT_INDEX.md`, `CHD_RM_EXPERIMENT_INDEX.md`, the route card, compact evidence README, and supersession note for the old process-failed A1X route.

## Decision

Fill only after terminal evidence is available.

- Verdict:
- Primary metric/gate reason:
- Mechanism/control reason:
- Preservation/safety reason:
- Evidence independence reason:
- Cost/deployability reason:
- Authorized next action or terminal stop:
