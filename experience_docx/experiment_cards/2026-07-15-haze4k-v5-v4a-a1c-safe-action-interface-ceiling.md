# Haze4K v5 CHD-RM v4a-A1C Safe-Action Interface Ceiling

Date: 2026-07-15

Status: `PLANNED`

## Scope

- Project: ConvIR-B Haze4K CHD-RM v5.
- Dataset or task: A1R fresh512 train-derived names, indices 256:768 of the frozen v3j controller-train list; no A1F names, A1X names, controller calibration, confirmation, canary, or locked test.
- Primary objective: determine whether the failed A1R frozen probe is principally limited by its half-resolution safe-action interface rather than by absence of recoverable action value.
- Main metric: paired-image bootstrap LCB95 of worst-operator interface headroom beyond privileged shrink on the full fresh512 development screen.
- Secondary metrics: full-reference headroom, native-size interface-retention ratio, repairable fraction, direct safety replay, support and bound excess, severe and hard regression counts, interface reconstruction error, wall time, and peak GPU memory.
- Execution environment: local WSL for editing and static checks only; runtime only on convir-4090.
- GitHub rules commit: `d502a990fb8766d6b0b8e57ed913d1a2200297de`.
- GitHub route branch and source commit: `codex/haze4k-v5-v4a-a1c-safe-action-interface-ceiling-20260715` from `github/main@d502a990fb8766d6b0b8e57ed913d1a2200297de`; this is a diagnostic route, not a model-structure route.
- Cloud `REMOTE_REPO`: derived only by schema-v2 convir-ops from repo name `ConvIR-B-v4a-a1c` and workspace id `a1c20260715`.
- Cloud `RUN_ROOT`: `/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v4a_a1c_safe_action_interface_ceiling_20260715`.
- Cloud `EVID_STAGE`: schema-v2 derived remote repository `experience_docx/experiment_logs/haze4k_v5_chd_rm_v4a_a1c_safe_action_interface_ceiling_20260715`.
- Explicit cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.

## Agent Execution Routing

Use `MODEL_AGENT_COST_ROUTING_PROTOCOL.md` for qualifications and dispatch mechanics.

| Applicable scope | Task class | Minimum role | Boundary action |
| --- | --- | --- | --- |
| A1C estimand, interface hierarchy, gates, and terminal interpretation | `R3_SCIENTIFIC_AUTHORITY` | `frontier` | user-pinned Sol/xhigh for this R3 design; later terminal interpretation is a new frontier task |
| frozen contract runner and static preflight implementation | `R2_ENGINEERING_CONTROL` | `balanced` | one Terra engineering package after this commit |
| exact S0 and formal launch with receipt-bound closeout | `R1_BOUNDED_EXECUTION` | `fast` | one Luna package per mechanically authorized stage |
| healthy formal observation windows | `R0_READ_ONLY` | `fast` | one persistent Luna monitor task |
| unchanged-verdict route evidence archive | `R2_ENGINEERING_CONTROL` | `balanced` | one Terra archival package after R3 terminal review |

## Baseline Contract

- Baseline implementation: exact A1F privileged safe direction and shrink replay over the frozen v3z A0R r1 final state; A1R half interface is reconstructed exactly before any new interface is compared.
- Baseline checkpoint or initialization: A0R epoch16 update512 final state SHA-256 `ed0832f220996af3fd8e617b7d04d643dc6ca052a3603adee99d59e78fd1e125`; no learned A1C parameters or initialization exist.
- Evaluation entrypoint: planned tracked `experience_docx/tools/chd_rm_v4a_a1c_safe_action_interface_ceiling.py` through `experience_docx/tools/run_chd_rm_v4a_a1c.sh`.
- Training entrypoint: none: this is a privileged deterministic evaluation with no model, probe, policy, or adapter training.
- Dataset and split: exactly A1R fresh512, four native-size-aware groups of 128 for grouped bootstrap bookkeeping; all 512 are development screening only.
- Preprocessing and decoding: immutable v3z Haze4K load/pad, A0R state restore, D_ref and D_rep operators, support, channel bounds, float32 add-and-clamp renderer, A1F safe grid, and old `.25` predecessor.
- Metric implementation: A1F RGB MSE/PSNR and safe candidate selection with 4,000 paired image bootstrap draws; worst operator is selected inside every draw.
- Reproduced baseline result: A1F formal heldout safe direction-over-shrink worst-operator LCB95 is `+0.105475 dB`, with repairable-fraction LCB95 `0.6953125` and zero severe or hard regressions.
- Known reproduction gap: A1R exact-half behavior must reproduce its original bilinear down/up action-interface construction before any A1C result is valid; otherwise the route stops as engineering failure.
- Reference entrypoints that must remain stable: A1F target, grid, safe selector, bootstrap, and v3z/v3w operator helpers; A1R target construction and `F.interpolate(..., mode="bilinear", align_corners=False)` half interface.
- Checkpoint/export/resume contract: no checkpoint, candidate export, resume, selector, policy, or deployable artifact is produced.

## Most Valuable Attempt

- Why this is the highest-value next attempt: A1F proves safe action headroom exists while A1R proves true frozen-representation signal exists but loses material utility at the half-resolution action interface.
- Target failure or opportunity: distinguish action-interface bandwidth/reconstruction loss from a genuine lack of recoverable direction or value information.
- Cheap preflight evidence: exact A1R half replay, full and anti-aliased tensor-shape audit, native-size group inventory, zero action identity, and direct safety replay on the fixed S0 names.
- Earliest decisive gate: whether the exact-half interface has worst-operator interface headroom LCB95 at least `+0.080 dB` and retains at least `0.50` of full-reference headroom in the worst native-size group.
- Expected cost or attempt-count saving: one deterministic three-interface diagnostic removes the need to guess among new model, probe, action, or policy variants.
- What success decides: the highest passing predeclared interface can be carried only to a separate A1X accessibility-and-deployability contract design.
- What failure decides: an interface with UCB95 below `+0.020 dB` is closed; if full also fails, the bounded safe-repair family closes.
- Why a cheaper diagnostic is not enough: tensor reconstruction error alone cannot establish safe action value under both operators and the fixed line-search safety contract.

## Hypothesis

- Observed failure: A1R true targets beat shuffled controls but its context-spatial half-resolution probe has only `+0.015205 dB` lift LCB95 and `0.092305` oracle-retention LCB95.
- Target mechanism: downsample-plus-upsample reconstruction of the action direction removes enough spatial amplitude and support detail to erase materially safe replay headroom.
- Null hypothesis: full, exact-half, and anti-aliased-half privileged interfaces have no material difference beyond shrink under the fixed action grid.
- Preferred causal hypothesis: conditional safe-action interface bandwidth and reconstruction are the active reachable-value bottleneck.
- Competing hypothesis or confound: full privileged advantage itself does not transfer to fresh512, or apparent differences arise from a changed support, grid, renderer, or source state.
- Cheapest observation that separates them: paired safe replay of the same privileged full direction after each frozen interface transformation, including an exact A1R half replay.

## Estimand And Risk Attribution

- Target population: the fixed 512 A1R fresh train-derived image names under D_ref and D_rep, interpreted only as development screening.
- Analysis unit and grouping unit: paired image/operator safe-replay row; image is resampled jointly across interfaces and operators, with native image-size stratum retained for the retention gate.
- Intervention or factor contrast: `H_interface = safe(full privileged direction) - safe(interface-reconstructed privileged direction)` and `H_full = safe(full privileged direction) - safe(shrink)`.
- Reference/direct predecessor: per-image privileged shrink selected on the unchanged A1F grid; the historical old `.25` action remains the direct predecessor for harm accounting.
- Outcome, direction, and aggregation: PSNR dB; positive headroom favors full over interface; mean within bootstrap draw then worst operator, and separately worst native-size ratio.
- Claim type: causal
- Identification assumptions and sensitivity limits: exact common source, full direction, action grid, renderer, and safety selector make interface assignment deterministic; the claim does not identify deployable prediction quality.
- Minimum worthwhile effect or risk limit: interface headroom worst-operator LCB95 `+0.080 dB`, worst-size interface/full ratio LCB95 `0.50`, and zero severe or hard regressions.
- Equivalence/non-inferiority margin and independent source: interface closure uses preregistered UCB95 below `+0.020 dB`, a conservative fraction of the A1R material-lift floor rather than a value derived from A1C results.
- Common safety anchor: unchanged A1F anchor, support mask, channel bounds, tanh/clamp construction, A1F safe grid, and float32 renderer.
- Inherited-harm estimand: old `.25` versus anchor burden is recorded unchanged per image/operator.
- Candidate-total-harm estimand: each interface-selected action versus the common anchor must remain nonworse under the frozen tolerance.
- Intervention-added-harm estimand: each interface-selected action versus old `.25` must have zero severe and zero hard regressions.

## Design And Identifiability

- Design type: `hybrid: paired privileged-interface ablation with hierarchical screening`.
- Why this is the cheapest design that identifies the estimand: the same full privileged direction is rendered through only the interface transformation, isolating interface loss without training or selection variance.
- Experimental unit and randomization/pairing: deterministic paired replay within image, operator, interface, and A1F grid; no random assignment and no learned state.
- Blocking, exclusion, failure, and missing-cell policy: native shape blocks are fixed before runs; any missing row, interface identity failure, nonfinite value, safety violation, or incomplete bootstrap family is engineering failure and no scientific gate is interpreted.
- Formal subgroup definitions and pre-intervention/independent source: native `400x400` and `480x640` shapes from the immutable A1R fresh manifest, defined before A1C execution.
- Primary comparison family and multiplicity treatment: ordered family exact-half, anti-aliased-half, full reference; evaluate each with simultaneous 95 percent bootstrap bounds across both operators and both native-size groups, advancing only after the previous interface passes.
- Fractional-design resolution and alias structure: not applicable: three deterministic interface levels share one privileged direction and no fractional factorial aliasing is used.
- Negligible-interaction assumptions and targeted de-alias follow-up: not applicable: both operators and size strata are retained explicitly; an interface-by-operator or interface-by-size failure blocks promotion.
- Paired seeds/folds/data order/evaluation operators: no training seed or fold; fixed sorted A1R names, D_ref and D_rep paired for every image, and bootstrap seed `3407` with 4,000 draws.
- Natural groups and repeated grouped-split or leave-one-group-out plan: image is the natural independent group; bootstrap samples images jointly and calculates the worst operator inside each draw; native-size ratio is evaluated by its prespecified stratum.
- Split/seed uncertainty required for the claim: fixed-data interval uncertainty only; no cross-training-seed generalization is claimed.
- Uncertainty estimator and dependence/group structure: paired nonparametric image bootstrap, 4,000 draws, simultaneous lower and upper 95 percent bounds over the preregistered interface and size family.
- Sample/group/split/seed count justified by power or target interval width: 512 images and two native-size strata match the completed A1R fresh screen and give 4,000-draw stable decision intervals without consuming A1X names.
- Fixed-data attainable precision or smallest reliably detectable effect: the `+0.080 dB` and `0.50` lower-bound gates are the smallest actionable interface loss for A1X design; `+0.020 dB` UCB closure prevents reopening a negligible interface.

## Change

- Code branch: `codex/haze4k-v5-v4a-a1c-safe-action-interface-ceiling-20260715`.
- Exact code/config change: new deterministic A1C evaluator will form full, A1R-exact bilinear half, and anti-aliased half privileged action interfaces, then apply the unchanged A1F safe grid.
- Enabled mechanisms: full reference, exact bilinear down/up replay, anti-aliased half down/up replay, native-size tracking, paired bootstrap, and hierarchical interface selection.
- Explicitly disabled mechanisms: model updates, probe fitting, feature changes, target learning, action selection from deployable information, policy fitting, candidate selection, canary, confirmation, and locked test.
- Parameter/runtime/memory impact expected: no trainable parameters; three deterministic replay interfaces, expected formal runtime below A1R formal and memory below 2 GiB excluding frozen model load.
- Initialization or no-op behavior: zero action is exactly the A1F/A1R replayed predecessor tensor; every interface must reproduce it with zero tensor discrepancy before formal work.
- Resume policy: none; a nonterminal run is not resumed and receives a fresh output id only after engineering review.
- Defaults changed: none outside this new evaluator.
- Defaults intentionally preserved: source state, assets, split, operators, support, bounds, grid, safe selector, renderer, bootstrap seed, and evidence-role restrictions.

## Preflight

| Check | Required evidence | Failure action |
| --- | --- | --- |
| source and A1R split | SHA identities, 512 names, exact native-size blocks, no A1F/A1X overlap | engineering stop |
| exact-half reconstruction | bitwise or documented zero-tolerance match to A1R bilinear interface | engineering stop |
| interface and zero action | tensor shapes, support, bounds, no-op MSE and PSNR identities | engineering stop |
| safety and statistics | complete paired rows, finite values, grouped bootstrap schema and simultaneous family | engineering stop |
| cloud route identity | fresh schema-v2 workspace, runner SHA, explicit Python, free output and session | preflight stop |

## Mechanism Metrics

- Primary metric: worst-operator `H_interface` LCB95 for the currently evaluated interface, measured full privileged safe replay minus same-direction interface safe replay.
- Interface retention metric: worst-native-size LCB95 of `H_interface / H_full`, where each bootstrap draw fails if the full-reference denominator is nonpositive.
- Reference metric: worst-operator full direction-over-shrink LCB95 validates that fresh512 retains a usable privileged reference.
- Reconstruction metric: per-interface directional cosine, normalized endpoint error, support mismatch, and exact-half match against the A1R construction are diagnostic only.
- Safety metric: exact anchor and predecessor nonworse counts, maximum bound/support excess, and severe/hard regression counts.

## Controls

| Control | Purpose | Required outcome |
| --- | --- | --- |
| full privileged direction | confirms A1F headroom transfers to the A1R fresh512 population | full reference is positive and safe |
| exact A1R bilinear half | isolates the actual A1R action-interface reconstruction | must match A1R construction before value is analyzed |
| anti-aliased half | separates sampling aliasing from reduced interface bandwidth | evaluated only after exact-half pass |
| zero and shrink actions | detect renderer/grid drift and retain common action reference | exact replay and safety identities |

## Evidence-Role Ledger

| Data or action | Role | Allowed use | Forbidden use |
| --- | --- | --- | --- |
| fixed S0 subset of fresh512 | engineering_debug | interface and runner integrity only | threshold tuning or selection |
| all A1R fresh512 | development_screening | hierarchical interface ceiling decision | confirmation, candidate, or policy claim |
| A1X remaining 432 names | confirmation | unavailable to A1C | any A1C read or cache |
| v3j controller calibration and route confirm | confirmation | unavailable | any A1C read |
| Haze4K test | sealed_final | unavailable | all A1C activity |

- Candidate/threshold/operator freeze point: all three interfaces, order, grid, thresholds, bootstrap seed, native-size groups, and operators freeze at this route commit before S0.
- Independent confirmation contract: only an A1C pass can authorize separate R3 design of A1X nested accessibility/deployability; A1C values cannot select or tune a deployable candidate.
- Nested group-respecting resampling contract: no A1C data can appear in a later confirmation fold; A1C bootstrap uses images jointly across interface/operator and keeps size strata fixed.
- Final sealed-use authorization and one-use policy: no sealed final authorization exists; Haze4K locked test remains blocked and one-use policy is reserved for a later explicit route.
- Post-sealed rule: not applicable: no sealed data is accessed; any later sealed access requires a separately frozen prior confirmation pass and no post-result adaptation.

## Fair Run Contract

- Training or inference budget: S0 is 16 fixed fresh512 names and formal is exactly 512 names times two operators times three interfaces; no training iterations.
- Batch/sample policy: native-shape blocks only, deterministic sorted names, no drop or replacement outside bootstrap resampling.
- Optimizer: not applicable: no parameters are optimized.
- Schedule: not applicable: one deterministic replay per interface and A1F grid.
- Loss weights: not applicable: no loss is optimized.
- Random seed policy: bootstrap seed `3407`; deterministic evaluator flags and fixed name order are recorded.
- Evaluation cadence: S0 writes one integrity summary; formal writes heartbeat after each native-size/operator block and summary after bootstrap.
- Checkpoint cadence: not applicable: no learned state exists.
- Hardware/runtime assumptions: one dynamically qualified convir-4090 CUDA GPU with at least 12000 MiB free and at most 10 percent utilization at launch; explicit cu121 Python.
- Allowed resume behavior: no resume; stop and diagnose any incomplete stage.
- Sample-size policy: S0 fixed 16 names spanning both native shapes; formal fixed fresh512 with no enlargement or substitution.
- Dependency/version assumptions: pinned route commit, A1F/A1R parent commits and asset hashes, PyTorch and NumPy from explicit cloud environment.
- Selected decision profile: `audit_evaluation`
- Learned-state retention required: no: no parameter, optimizer, policy, or learned state is created.
- Omitted or specialized stage rationale: deterministic interface ceiling is the necessary bridge before any accessibility, adapter, policy, confirmation, or locked-test route can be scientifically justified.
- Integrated pre-smoke contract: S0 in the tracked runner verifies source, names, interface reconstruction, zero/shrink replay, safety, finite values, and compact closeout before formal authorization.
- Expected wall-time budget and required phase timings: S0 under 20 minutes and formal under 90 minutes; runner records setup, replay, bootstrap, summary, and closeout timings.
- Heartbeat cadence, stale threshold, and monitor profile: heartbeat every 60 seconds, stale at 300 seconds, S0 `short` monitor and later formal `standard` monitor.
- Maximum model-visible observations and escalation condition: one start receipt and at most nine S0 or twenty-one formal status windows; stale heartbeat, output mismatch, or missing closeout escalates to R2.
- Workspace policy: fresh_route for S0; later formal may use exact_continuation only after a clean same-branch fast-forward check and typed S0 pass.

## Gates

| Stage | Estimand/question | Evidence role and scope | Gate type and threshold | `PASS` authorizes |
| --- | --- | --- | --- | --- |
| A1C S0 | Are source, three interfaces, exact A1R half replay, no-op, grid, safety, and statistics structurally valid? | engineering_debug, fixed 16 fresh512 names | all identities and rows complete; zero action exact; bound/support excess 0; severe/hard 0; exact-half match; finite paired bootstrap inputs | A1C formal only |
| A1C formal exact-half | Does exact A1R half interface lose material safe headroom? | development_screening, all fresh512 | worst-operator `H_interface` LCB95 `>=+0.080 dB`, worst-size ratio LCB95 `>=0.50`, full reference positive, zero severe/hard | anti-aliased-half formal only |
| A1C formal anti-aliased-half | Does anti-aliasing preserve a viable half interface? | development_screening, all fresh512 | same gates; if UCB95 `<+0.020 dB`, close this interface | full-reference formal only if half fails or closes |
| A1C formal full | Does full interface preserve viable privileged action headroom? | development_screening, all fresh512 | full reference positive and direct safety intact; if it fails, close bounded safe-repair family | R3 review only |
| A1X, adapter, confirmation, sealed final | not part of A1C | none | prohibited | none |

## Analysis Plan

- Per-sample or subgroup analysis: cloud-only paired rows by image, interface, operator, native shape, grid action, and safety diagnostics; compact interface/operator and size summaries only.
- Robustness or held-out analysis: fresh512 is development screening, not held-out confirmation; paired interface comparison, both operators, and two prespecified native-size strata are mandatory.
- Regression analysis: none; no fitted regression, calibration, or target learning is allowed.
- Main-effect/interaction and alias analysis: report interface-by-operator and interface-by-native-size contrasts descriptively; hierarchical gate order, not post hoc contrast ranking, controls continuation.
- Group/split/seed uncertainty and sensitivity analysis: 4,000 paired image bootstrap draws with worst operator and simultaneous interface/size family bounds; no seed-generalization claim.
- Screening-selection versus confirmation analysis: selection is limited to a technical interface contract for A1X design, never a model or deployment candidate; A1X remains untouched.
- Required docs to update: this card, A1C evidence README, route branch index entries at terminal handoff, typed closeouts, bootstrap summaries, and R3 review.
- Required artifacts to retain: runner, source manifest, interface specification manifest, compact operator/size/bootstrap summaries, closeout, status, and README.
- Required artifacts to delete or keep external: raw rows, rendered tensors, images, arrays, and logs remain under cloud RUN_ROOT; no large artifacts enter Git.
- Evidence package contents: route card, initial authorization, operations manifest, README, source manifest, interface summary, bootstrap summary, typed closeout, and R3 review.

## Decision

- Decision label: `V4A_A1C_PLANNED_SAFE_ACTION_INTERFACE_CEILING_ONLY`.
- Image/global metric reason: only paired image-level safe replay with within-draw worst operator can quantify interface headroom without pixel-count distortion.
- Mechanism reason: exact-half and anti-aliased-half isolate reconstruction and sampling from the same full privileged direction.
- Preservation or regression reason: all interface actions use the frozen A1F safe selector and must remain anchor- and predecessor-nonworse.
- Inherited harm versus anchor: old `.25` harm is retained as a read-only baseline diagnostic.
- Candidate total harm versus anchor: every selected interface action must be nonworse than the common anchor.
- Intervention-added harm versus predecessor: zero severe and hard regressions versus old `.25` are required for every interface.
- Group/split/seed uncertainty and interaction reason: paired image bootstrap, worst operator, and worst native-size retention protect the decisive dependence structure.
- Evidence role and independence reason: fresh512 is development screening only; neither A1X, confirmation, nor locked test is touched.
- Cost/deployability reason: privileged replay is an upper-bound diagnostic, not a deployable selector, policy, or model claim.
- What this decides next: only a full A1C pass selects an interface for separate R3 A1X nested accessibility-and-deployability design; both half failures restrict that design to full interface; full failure closes bounded repair.
- Typed closeout path: `RUN_ROOT/run-id/v4a_a1c_s0_closeout.json` for S0 and later stage-specific closeouts.
- `PASS` authorizes: `A1C_FORMAL_EXACT_HALF_ONLY` for S0; later formal transition is only the next listed hierarchical interface or R3 review.
- `INCONCLUSIVE` authorizes: `SAME_STAGE_ENGINEERING_REPAIR_ONLY`.
- `FAIL` stops: `DIRECT_TRAINING_POLICY_CANDIDATE_CANARY_LOCKED_TEST`.
