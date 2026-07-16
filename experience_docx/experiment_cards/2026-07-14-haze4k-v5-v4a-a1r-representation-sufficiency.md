# Haze4K v5 CHD-RM v4a-A1R Representation Sufficiency

Date: 2026-07-14

Status: `PLANNED`

## Scope

- Project: ConvIR-B Haze4K CHD-RM v5.
- Model family: diagnostic zero-init bounded `Delta-u` probes over frozen v3z output/context representations; no deployable candidate.
- Dataset or task: fresh train-derived names 256:768 from the frozen 1,200-name v3j list, grouped into four OOF folds; no A1F names, canary, or locked test.
- Primary objective: determine whether deployable frozen output/context information can recover a material fraction of the A1F safe direction headroom outside the failed local v3z head.
- Main metric: fresh512 OOF paired-image bootstrap LCB95 of the worst-operator mean safe direction-over-shrink lift for the primary `context_spatial` probe.
- Secondary metrics: A1F-oracle headroom retention, true-minus-shuffled lift, repairable fraction, output/context and linear/spatial factor contrasts, target endpoint MSE/cosine, safety excess, severe/hard counts, grid choice, runtime, and memory.
- Execution environment: local WSL for editing and static checks only; runtime only on `convir-4090`.
- GitHub rules commit: `42cefba8a1312d298e931ddac61a9eb4c917bab0`.
- Local WSL path, if used for editing/static checks: `/home/ubuntu/workspace/ConvIR-B-v4a-a1r-representation-sufficiency-20260714`.
- GitHub route branch and source commit: `codex/haze4k-v5-v4a-a1r-representation-sufficiency-20260714` from immutable `github/codex/haze4k-official-arch-anchor@3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`.
- Cloud `REMOTE_REPO`: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v4a-a1r-representation-sufficiency-20260714`.
- Cloud `RUN_ROOT`: `/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v4a_a1r_representation_sufficiency_20260714`.
- Cloud `EVID_STAGE`: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v4a-a1r-representation-sufficiency-20260714/experience_docx/experiment_logs/haze4k_v5_chd_rm_v4a_a1r_representation_sufficiency_20260714`.
- Explicit cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.

## Historical Execution Note

The retired multi-model execution packaging has no current authority and does
not change this card's scientific evidence. New work follows current GitHub
`main` rules and one qualified warm task.

## Baseline Contract

- Baseline implementation: immutable v3z projected output head source `3caddcc5265732e5be77e3404119a28cb28c11e6`, restored through the exact A0R r1 final state and A1F helpers.
- Baseline checkpoint or initialization: A0R epoch16/update512 final state SHA-256 `ed0832f220996af3fd8e617b7d04d643dc6ca052a3603adee99d59e78fd1e125`; every probe final layer is exactly zero-init.
- Evaluation entrypoint: `experience_docx/tools/chd_rm_v4a_a1r_representation_sufficiency.py` through the tracked stage runner.
- Training entrypoint: the same audit entrypoint trains only small diagnostic OOF probes on cached frozen maps; ConvIR-B and all historical heads remain frozen.
- Dataset and split: frozen 1,200-name train list; A1F used indices 0:256, A1R uses exactly indices 256:768 with deterministic four-fold grouped assignment.
- Preprocessing and decoding: immutable v3z Haze4K load/pad, base/control/gate, D_ref/D_rep steps, old support/bounds, and float32 add-and-clamp renderer.
- Metric implementation: exact A1F shrink and GT-direction grids, safe candidate selection, RGB MSE/PSNR, and 4,000 paired image bootstrap draws; A1R adds predicted-direction cells.
- Reproduced baseline result: A1F formal exact replay passed and established heldout worst-operator direction-over-shrink LCB95 `+0.105475 dB` with repairable-fraction LCB95 `0.6953125`.
- Known reproduction gap: none accepted for pinned source/state/assets; A1R smoke fails closed on source, zero-init, finite-gradient, fold, shape, grid, or forbidden-data mismatch.
- Reference entrypoints that must remain stable: v3z `candidate_predictions`, v3w frozen sample/operator path, A1F target/grid/safety helpers, A0R final state, and v3j split/bounds.
- Checkpoint/export/resume contract: retain final diagnostic probe/normalizer states and hashes only under `RUN_ROOT`; no resume, deployable export, or checkpoint selection.

## Most Valuable Attempt

- Why this is the highest-value next attempt: A1F proves a safe target direction exists, while v4a excludes the tested local optimizer/projection/window corrections; the next decision is whether current deployable information can represent that direction.
- Target failure or opportunity: separate missing information/context from insufficient local spatial readout and from the prior safety/render objective.
- Cheap preflight evidence: fresh32 exact source replay, disjoint-name/fold audit, zero-init current-state equivalence, finite target loss/gradient, and target-shuffle identity.
- Earliest decisive gate: primary context-spatial OOF direction must materially exceed shrink and its shuffled control while retaining a preregistered fraction of the GT oracle.
- Expected cost or attempt-count saving: one frozen feature cache plus small probes can stop or select the representation class before any full model, candidate, or safety training.
- What success decides: existing frozen full-context representation with local spatial readout is sufficient for a separately preregistered target-learning training-contract design.
- What failure decides: current output/context representations and local probe family are insufficient; direct training is stopped and only materially new representation design may be considered.
- Why a cheaper diagnostic is not enough: A1F uses GT directly and v3t/v3z optimize renderer objectives; neither measures fresh-name OOF target-direction information against a shuffled mapping control.

## Hypothesis

- Observed failure: the local v3z head cannot reach heldout safety, although A1F finds broad safe direction headroom beyond shrink.
- Target mechanism: frozen full-context maps contain spatial direction information absent or hard to use in the output tuple, and a local spatial probe can recover it under direct target supervision.
- Null hypothesis: primary context-spatial OOF lift is below the worthwhile effect, retains too little oracle headroom, or is not better than shuffled target mapping.
- Preferred causal hypothesis: assigned representation/probe factors change fresh-name OOF safe direction recovery because context and local spatial mixing expose target-aligned information.
- Competing hypothesis or confound: privileged safe line search, generic population-average direction, capacity, or image leakage creates apparent value without true input-target mapping.
- Cheapest observation that separates them: paired 2x2 OOF probes plus the fixed context-spatial cross-image target-shuffle control on disjoint names.

Mechanism sentence:

```text
On fresh grouped OOF images, a zero-init local spatial readout of frozen full
context should retain material A1F safe direction headroom beyond shrink and
beat a cross-image shuffled target control if the next bottleneck is the prior
head/objective rather than missing deployable representation information.
```

## Estimand And Risk Attribution

- Target population: frozen train-derived names 256:768, disjoint from all A1F update/heldout names; inference is limited to this fixed development population.
- Analysis unit and grouping unit: clean-reference image, with D_ref/D_rep and all probe cells paired within image.
- Intervention or factor contrast: output12 versus frozen-context+output12 representation, crossed with linear versus local-spatial readout; primary cell is context-spatial, with a cross-image shuffled-target control.
- Reference/direct predecessor: privileged shrink oracle is the primary reference; exact GT-direction oracle supplies the attainable denominator; current v3z and old `.25` are diagnostics.
- Outcome, direction, and aggregation: higher safe rendered PSNR is better; worst-operator mean OOF lift and bootstrap lower bounds decide.
- Claim type: causal
- Identification assumptions and sensitivity limits: factors are assigned with identical names/folds/targets/budgets and only the representation/readout changes; privileged line search makes the result nondeployable and estimates direction information, not direct action calibration.
- Minimum worthwhile effect or risk limit: primary worst-operator LCB95 over shrink `>=+0.020 dB`, oracle-retention LCB95 `>=0.25`, true-minus-shuffle LCB95 `>=+0.005 dB`, and repairable-fraction LCB95 `>=0.20`.
- Equivalence/non-inferiority margin and independent source: `+0.020 dB` is the pre-A1R v3s formal utility floor; `+0.005 dB` is the pre-A1R v3r SESOI; `0.20` is the pre-A1R repair floor; retention `0.25` requires a material quarter of parent A1F headroom.
- Common safety anchor: old `.125` for each fresh image/operator.
- Inherited-harm estimand: `max(MSE(old_.25)-MSE(old_.125),0)`.
- Candidate-total-harm estimand: `max(MSE(selected_probe_.25)-MSE(old_.125),0)` under privileged safe selection.
- Intervention-added-harm estimand: candidate total harm minus inherited harm, with every selected action additionally predecessor-nonworse.

## Design And Identifiability

- Design type: `full_factorial`
- Why this is the cheapest design that identifies the estimand: frozen maps are cached once and small probes isolate representation, local spatial capacity, and mapping validity before any full image model training.
- Experimental unit and randomization/pairing: fixed image groups; all cells/operators share folds, cache, target, optimizer schedule, and evaluation draws; exact spatial size is a nuisance block for batching and the target shuffle is a deterministic within-operator-and-size-block cyclic permutation of training images.
- Blocking, exclusion, failure, and missing-cell policy: four fixed OOF folds block image identity; no row is excluded; any missing fold/cell/name/operator, nonfinite value, state/hash mismatch, or unsafe selected row fails closed.
- Formal subgroup definitions and pre-intervention/independent source: representation, readout, operator, and fresh fold only; A1F harm strata may be descriptive but cannot alter gates.
- Primary comparison family and multiplicity treatment: one preregistered primary context-spatial gate; other 2x2 cells are paired mechanism controls and cannot independently authorize continuation.
- Fractional-design resolution and alias structure: full 2x2 representation-by-readout design has no aliasing; shuffled context-spatial is a separate negative control.
- Negligible-interaction assumptions and targeted de-alias follow-up: no representation-by-readout interaction is assumed negligible; all four true-target cells are complete and reported.
- Paired seeds/folds/data order/evaluation operators: seed `3407`, identical four folds, deterministic batch order, and paired D_ref/D_rep for every cell.
- Natural groups and repeated grouped-split or leave-one-group-out plan: four-fold OOF assigns each clean-reference name to exactly one heldout fold and trains only on the other 384 names.
- Split/seed uncertainty required for the claim: complete OOF predictions and fold summaries are required; one fixed optimization seed limits the claim to representation screening, not algorithmic seed robustness.
- Uncertainty estimator and dependence/group structure: 4,000 PCG64(3407) image bootstrap draws resample complete paired cell/operator rows; the worst operator is computed within each draw.
- Sample/group/split/seed count justified by power or target interval width: fresh512 provides four 128-image heldout folds and is four times the A1F decision population while staying inside unused train-derived evidence.
- Fixed-data attainable precision or smallest reliably detectable effect: the observed grouped-bootstrap intervals determine precision; failure to clear the fixed lower bounds is a screening fail, not an equivalence claim.

Factor family:

| Cell | Representation | Readout | Target mapping | Role |
| --- | --- | --- | --- | --- |
| `output_linear` | downsampled hazy/base/old-step/current-Delta-u | zero-init 1x1 linear | true | capacity floor |
| `output_spatial` | same output12 | zero-init 3x3-depthwise-1x1 local head | true | output information control |
| `context_linear` | frozen full context plus output12 | zero-init 1x1 linear | true | context linear separability |
| `context_spatial` | frozen full context plus output12 | zero-init local spatial head | true | preregistered primary |
| `context_spatial_shuffled` | same as primary | same local spatial head | cyclic cross-image target permutation within operator/exact-size/train-fold block | mapping control |

## Adaptive Decision Paths

| Frozen trigger | Authorized next branch | Budget/data role | Stops or forbids |
| --- | --- | --- | --- |
| smoke source/no-op/gradient integrity pass | full fresh512 four-fold OOF A1R | `development_screening` | no direct candidate or sealed data |
| primary formal gate passes and true beats shuffle | R3 design of a separate target-learning training contract only | new untouched confirmation contract required | no direct reuse of A1R outcomes for confirmation |
| primary fails but a nonprimary control appears positive | R3 interpretation/amendment only | A1R development evidence | no automatic cell selection or training |
| structural/control/primary gate fails | terminal A1R closeout or same-stage engineering repair by failure class | engineering or development only | direct training, policy, canary, locked test |

## Change

- Code branch: `codex/haze4k-v5-v4a-a1r-representation-sufficiency-20260714`.
- Exact code/config change: add a no-deployment probe audit that restores the exact v3z state, constructs the fixed A1F target on fresh names, trains five small OOF cells, and evaluates predicted direction through the unchanged safe grid. After smoke r1 exposed the two native Haze4K spatial sizes, batch construction and the shuffled control were amended before any probe update to block on exact feature size without resize, crop, or padding.
- Enabled mechanisms: fold-specific feature normalization, direct bounded target-endpoint regression, 2x2 representation/readout factors, shuffled target control, paired safe-grid replay, and grouped bootstrap.
- Explicitly disabled mechanisms: ConvIR/control/operator/current-head updates, renderer/safety loss tuning, action-bound/support expansion, selector/threshold/policy fitting, candidate selection, canary, locked test, and architecture promotion.
- Parameter/runtime/memory impact expected: diagnostic probes only, final states under `RUN_ROOT`; one frozen cache/evaluation pass per image plus 20 small fold/cell fits.
- Initialization or no-op behavior: zero probe correction exactly returns the current v3z `Delta-u` endpoint before any update.
- Resume policy: none; interrupted smoke/formal restarts under a fresh run id and output directory.
- Defaults changed: direct normalized endpoint target replaces historical renderer/safety objectives only inside the diagnostic probe fits.
- Defaults intentionally preserved: frozen assets/state, name order, operators, support, RGB bounds, renderer, action grids, anchor, seed, and metric code.

## Preflight

| Check | Pass line | Result |
| --- | --- | --- |
| route identity | fresh anchor branch, exact route commit, clean local/cloud checkout | local complete; cloud required |
| parent authorization | A1F R3 review SHA `a8b9064308710ac5fc890b9de0158c1faddb4d51f7d298d4991e9ddfb3616e1d` authorizes `A1R_ROUTE_DESIGN_ONLY` | source complete; runtime recheck required |
| state/source identity | A0R final state, v3z/A1F commits, source/asset hashes exact | runtime recheck required |
| data isolation | exact fresh indices 256:768, zero overlap with A1F 0:256, four folds of 128 | static complete; runtime recheck required |
| factor contract | five complete cells, true/shuffled mapping identities within exact spatial-size blocks, zero-init current-state equality | static complete; smoke required |
| GPU capacity | selected GPU has at least 18,000 MiB free and at most 10% utilization | dynamic check required |

## Mechanism Metrics

| Metric | Why it matches the route | Gate subset | Compact artifact |
| --- | --- | --- | --- |
| safe predicted direction over shrink | direct value beyond abstention/amplitude | fresh512 OOF, worst operator | bootstrap summary |
| GT-oracle headroom retention | fraction of attainable A1F direction captured | fresh512 OOF, worst operator | bootstrap summary |
| true-minus-shuffled lift | rejects generic/population-average direction | primary versus paired shuffled control | bootstrap summary |
| active target endpoint MSE/cosine | measures representation/readout alignment | all OOF cells | cell/operator summary |
| repairable fraction and safety excess | requires broad safe value without inherited-risk confusion | all OOF cells | summary and closeout |

## Controls

| Control | Purpose | Pass line |
| --- | --- | --- |
| exact frozen source/current state | exclude A1F/v3z drift | pinned commits/hashes and smoke replay exact |
| zero-init current endpoint | validate no-op semantics | max tensor discrepancy `0.0` for every cell |
| output/context x linear/spatial matrix | separate representation from local readout | all four true cells complete under matched budget |
| context-spatial shuffled target | reject image-independent target prior or leakage | fixed cyclic mapping, no self-pair, primary true-minus-shuffle gate |
| exact GT direction oracle | denominator and fresh-population ceiling | same A1F target/grid/safety code, both operators |

## Evidence-Role Ledger

| Evidence source or groups | Role | Allowed uses | Forbidden uses |
| --- | --- | --- | --- |
| first 32 fresh names smoke | `engineering_debug` | source, fold, native-size block, no-op, finite gradient, mapping checks | scientific threshold or cell selection |
| fresh512 four-fold OOF | `development_screening` | primary representation decision and mechanism controls | promotion or independent confirmation |
| A1F 0:256 names/results | historical parent evidence | thresholds and target contract only | A1R training, validation, or rescue |
| Haze4K locked test | `sealed_final` | none in A1R | all access |

- Candidate/threshold/operator freeze point: factor cells, target, folds, optimizer, budgets, metrics, thresholds, and operators freeze in this card before smoke.
- Independent confirmation contract: A1R has none; a pass authorizes only design using names/evidence not consumed by A1R and a separately frozen direct-action contract.
- Nested group-respecting resampling contract, if no separate confirmation set: not used for a promotion claim; outer OOF rows support development screening only.
- Final sealed-use authorization and one-use policy: no sealed use is authorized.
- Post-sealed rule (`report/close only; no tuning or reselection`): not applicable because sealed data remain untouched.

## Fair Run Contract

- Training or inference budget: smoke caches 32 names and performs two diagnostic updates per cell; formal caches 512 names and trains five cells across four OOF folds for eight epochs each.
- Batch/sample policy: native action-resolution tensors, deterministic exact-spatial-size blocks and name/operator order, batch size at most eight, both operators as paired items, no resize/crop/padding, exclusions, substitutions, or augmentation.
- Optimizer: AdamW LR `5e-4`, weight decay `1e-5`, gradient clip `0.1`, independently reset for every fold/cell.
- Schedule: constant LR, eight formal epochs, no early stopping or checkpoint selection.
- Loss weights: one normalized active-support endpoint MSE only; shuffled cell changes only target-name assignment.
- Random seed policy: seed `3407` for initialization/order/bootstrap; deterministic four-fold and cyclic shuffle mappings.
- Evaluation cadence: final OOF state only; epoch loss is engineering history, not selection.
- Checkpoint cadence: final probe plus normalizer per fold/cell only; no intermediate selection.
- Hardware/runtime assumptions: `convir-4090`, explicit cu121 Python, one dynamically qualified CUDA GPU.
- Allowed resume behavior: none.
- Sample-size policy: smoke 32; formal exactly 512 unique fresh images, four heldout folds of 128, two operators, five cells.
- Dependency/version assumptions: pinned A1F/v3z/v3s/v3p commits and all source/state/asset hashes; PyTorch/NumPy from explicit environment.
- Selected decision profile: `factorial_screening`
- Learned-state retention required: yes
- Omitted or specialized stage rationale: no full model training, direct-action confirmation, policy replay, canary, or locked test can answer the representation-information question.

## Learned-State Retention

- Retained steps/epochs/factor cells and why each is needed: final epoch for all 20 formal fold/cell fits; complete family is required to reproduce OOF predictions and factor controls.
- Model/checkpoint state path and hash contract: `RUN_ROOT/run-id/models/cell/foldN.pt` plus SHA-256 in `probe_state_manifest.json`; raw states remain cloud-only.
- Optimizer/scheduler state contract: no resume/dynamics claim, so optimizer state is not retained; fixed optimizer/schedule and final model state are sufficient for OOF reconstruction.
- RNG states required and unavailable-state disclosure: seed, deterministic flags, fold map, batch order, shuffle map, and PyTorch initial seed are recorded; accelerator RNG snapshot is not required after final deterministic state save.
- Data-order/sampler identity: exact-size block then sorted fresh name/operator, recorded four-fold assignment, deterministic batch slices, no stochastic sampler; shuffle maps are cyclic only within operator and exact-size block.
- Config hash, code commit, Python/environment identity, and parent checkpoint: source manifest records route commit/card SHA, explicit Python, probe config hash, parent source commits, A0R state SHA, and asset hashes.
- Trace-manifest path and schema: `RUN_ROOT/run-id/probe_state_manifest.json`, schema version 1 with cell/fold/path/hash/train/heldout counts and normalizer shapes.
- Cloud retention/deletion policy: final states, raw OOF rows, cache diagnostics, histories, logs, and manifests remain under `RUN_ROOT`; in-memory feature caches are released after the run.
- Compact GitHub evidence: typed closeout, source/state manifests without tensors, cell/operator summary, bootstrap summary, fold history summary, README, and R3 review only.

## Gates

| Stage | Estimand/question | Evidence role and scope | Gate type and threshold | `PASS` authorizes |
| --- | --- | --- | --- | --- |
| S0 smoke | Are source, fresh split, native-size blocks, target, cells, no-op, and gradients structurally valid? | `engineering_debug`, first 32 fresh names | exact identities/folds/no overlap; every size block has at least two names; zero endpoint discrepancy; all five cells finite with nonzero gradient; size-blocked shuffle is complete and has no self-pair | A1R formal only |
| A1R formal | Does primary frozen context plus local spatial readout contain material target direction information? | `development_screening`, full fresh512 four-fold OOF | 4,000 paired draws; worst-operator primary over-shrink LCB95 `>=+0.020 dB`, oracle-retention LCB95 `>=0.25`, true-minus-shuffle LCB95 `>=+0.005 dB`, repairable LCB95 `>=0.20`, complete family and pointwise safety | R3 handoff for separate target-learning training-contract design only |
| independent confirmation | not part of A1R | none | prohibited | none |
| sealed final | not part of A1R | none | prohibited | none |

## Analysis Plan

- Per-sample or subgroup analysis: cloud-only OOF rows by cell/fold/operator; compact cell/operator means, p05/p10, target loss/cosine, repairable fraction, grid choice, and safety excess.
- Visual or qualitative analysis: none required; image outputs are not retained as compact evidence.
- Complexity analysis: report probe parameter counts, wall time, and peak allocated GPU memory; no deployment-cost claim.
- Robustness or held-out analysis: every fresh image is held out exactly once; update/A1F names cannot rescue failure; both operators are paired and worst decides.
- Regression analysis: direct normalized endpoint regression is the assigned probe objective; no post-result loss or calibration regression is allowed.
- Main-effect/interaction and alias analysis, if applicable: report paired representation, readout, and interaction contrasts from the complete 2x2 true-target family; primary authorization remains context-spatial only.
- Group/split/seed uncertainty and sensitivity analysis: paired image bootstrap over complete OOF rows and per-fold summaries; no multi-seed generalization claim.
- Screening-selection versus confirmation analysis: controls diagnose the bottleneck but cannot be selected post hoc as a candidate; any next representation/action contract requires new preregistration and untouched evidence.
- Required docs to update: this card, evidence README, CHD-RM index, global index, typed closeout, and R3 review at terminal handoff.
- Required artifacts to retain: runner, source manifest, probe-state manifest/hashes, compact histories, cell/operator/bootstrap summaries, closeout, and status.
- Required artifacts to delete or keep external: probe tensors, raw OOF rows, caches, full logs, images, arrays, and final state files remain cloud-only.
- Evidence package contents: card, README, source/state summaries, closeout, fold history, cell/operator summary, bootstrap summary, and R3 review.

## Decision

- Decision label: `V4A_A1R_PLANNED_FRESH_OOF_REPRESENTATION_SUFFICIENCY_ONLY`.
- Image/global metric reason: complete fresh-name OOF paired image inference is required; training loss or pixel count cannot decide representation value.
- Mechanism reason: the 2x2 cells separate frozen context information from local spatial readout, and shuffle tests true mapping.
- Preservation or regression reason: privileged evaluation selects only anchor- and predecessor-nonworse actions before aggregate value is measured.
- Inherited harm versus anchor: old `.25` burden remains explicit per row.
- Candidate total harm versus anchor: selected probe direction uses the same old `.125` anchor.
- Intervention-added harm versus predecessor: selected probe action cannot exceed old `.25` beyond frozen numerical tolerance.
- Group/split/seed uncertainty and interaction reason: four OOF folds, paired operators/cells, and within-draw worst operator retain dependence; single seed limits the claim.
- Evidence role and independence reason: A1R is development screening and cannot confirm or promote a representation.
- Cost/deployability reason: small probes and privileged line search answer information sufficiency cheaply but are not an inference policy.
- What this decides next: primary pass permits R3 target-learning training-contract design only; failure closes current output/context local probe family.
- Typed closeout path: `RUN_ROOT/run-id/v4a_a1r_closeout.json`.
- `PASS` authorizes: `R3_REVIEW_FOR_TARGET_LEARNING_CONTRACT_DESIGN_ONLY`.
- `INCONCLUSIVE` authorizes: `SAME_STAGE_ENGINEERING_OR_PREDECLARED_EVIDENCE_REPAIR_ONLY`.
- `FAIL` stops: `DIRECT_TRAINING_POLICY_CANDIDATE_CANARY_AND_LOCKED_TEST`.
