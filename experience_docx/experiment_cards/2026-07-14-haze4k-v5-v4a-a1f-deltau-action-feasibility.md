# Haze4K v5 CHD-RM v4a-A1F Delta-u Action Feasibility

Date: 2026-07-14

Status: `COMPLETED`

## Scope

- Project: ConvIR-B Haze4K CHD-RM v5.
- Model family: frozen v3z output-side Delta-u repair head at the exact A0R r1 final state; privileged action audit only.
- Dataset or task: exact v3z update128 and disjoint heldout128 train-derived names; no canary or locked-test access.
- Primary objective: determine whether the v3z-aligned bounded Delta-u action space contains safe direction-changing headroom beyond privileged shrink/abstention.
- Main metric: heldout128 paired-image bootstrap LCB95 of the worst-operator mean PSNR lift of the safe direction oracle over the safe shrink oracle.
- Secondary metrics: lift versus current v3z and old `.25`, repairable-image fraction, inherited/total/added harm, p05/p10 lift, severe/hard regressions, action-grid choice, and exact A0D replay discrepancy.
- Execution environment: local WSL for editing and static checks only; runtime only on `convir-4090`.
- GitHub rules commit: `12dcac637f34354aeab5c28a9d93d10adb94f98d`.
- Local WSL path, if used for editing/static checks: `/home/ubuntu/workspace/ConvIR-B-v4a-a1f-deltau-action-feasibility-20260714`.
- GitHub route branch and source commit: `codex/haze4k-v5-v4a-a1f-deltau-action-feasibility-20260714` from v4a handoff `8d3ae09e382bac4daf3968ee511ed14c0316da39`.
- Cloud `REMOTE_REPO`: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v4a-a1f-deltau-action-feasibility-20260714`.
- Cloud `RUN_ROOT`: `/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v4a_a1f_deltau_action_feasibility_20260714`.
- Cloud `EVID_STAGE`: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v4a-a1f-deltau-action-feasibility-20260714/experience_docx/experiment_logs/haze4k_v5_chd_rm_v4a_a1f_deltau_action_feasibility_20260714`.
- Explicit cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.

## Historical Execution Note

The retired multi-model execution packaging has no current authority and does
not change this card's scientific evidence. New work follows current GitHub
`main` rules and one qualified warm task.

## Baseline Contract

- Baseline implementation: immutable v3z projected-head source at `3caddcc5265732e5be77e3404119a28cb28c11e6`, reconstructed by v4a A0R.
- Baseline checkpoint or initialization: A0R r1 final state `epoch16_update512_final.pt`, SHA-256 `ed0832f220996af3fd8e617b7d04d643dc6ca052a3603adee99d59e78fd1e125`.
- Evaluation entrypoint: `experience_docx/tools/chd_rm_v4a_a1f_action_feasibility.py` through the tracked stage runner.
- Training entrypoint: none; training and optimizer updates are forbidden.
- Dataset and split: the first 128 `v3j_controller_train` names are update128 and the next 128 disjoint names are heldout128, with the exact A0R fold map.
- Preprocessing and decoding: immutable v3z frozen sample path, D_ref/D_rep operators, old support, `.125/.25` add-and-clamp renderer, and canonical per-image RGB MSE.
- Metric implementation: replay old `.125`, old `.25`, and current v3z `.25` against A0D rows; evaluate all new actions through the same float32 add-and-clamp tensors and aggregate in float64.
- Reproduced baseline result: A0R exact no-op and historical trajectories have maximum discrepancy `0.0`; A0D has 512 finite rows; A0P completed 2,304 structurally valid cells with no local correction.
- Known reproduction gap: none is accepted; A1F fails integrity if any of the 512 formal A0D rows differs by more than `1e-12` MSE or `1e-9 dB`.
- Reference entrypoints that must remain stable: v3z `candidate_predictions`, v3w frozen sample/operator path, A0R r1 trace manifest, A0D raw row table, and v3j action bounds.
- Checkpoint/export/resume contract: no resume and no learned export; an interrupted run restarts under a new run id and fresh output directory.

## Most Valuable Attempt

- Why this is the highest-value next attempt: A0P ruled out the tested optimizer, projection-order, and risk-window repairs, so the next branch depends on whether safe bounded actions exist at all under the exact failed contract.
- Target failure or opportunity: v3z current-head utility is positive, but heldout anchor/harm fail; privileged direction repair had large historical ceiling under v3r but was not aligned to the v3z final state and v4a risk contract.
- Cheap preflight evidence: exact A0D per-image replay from the one A0R final state on a fixed 8+8 smoke subset.
- Earliest decisive gate: direction-line headroom must exceed privileged shrink/abstention by the independently sourced `+0.005 dB` worst-operator LCB and `20%` repairable-image fraction.
- Expected cost or attempt-count saving: one inference pass per image plus tensor-only fixed grids can stop all representation work before any training.
- What success decides: only that a separately preregistered representation-sufficiency audit is informative.
- What failure decides: terminal closeout of this learned Delta-u repair family under the current bounds/support/anchor contract.
- Why a cheaper diagnostic is not enough: v3r establishes generic old-operator direction ceiling, while A1F must align the v3z final repair, heldout128 failure population, and v4a total/incremental safety definitions.

## Hypothesis

- Observed failure: exact and proposal-space local parameter updates do not improve v3z heldout safety under any frozen risk window.
- Target mechanism: a bounded spatial Delta-u direction may remain safe and useful even though the current low-capacity head and its local parameter Jacobian cannot reach it.
- Null hypothesis: after privileged shrink/abstention is allowed, the v3z-aligned bounded direction family adds less than the minimum worthwhile effect or reaches too few heldout images.
- Preferred causal hypothesis: the augmented direction action set produces material safe headroom beyond shrink, favoring representation insufficiency over action-space infeasibility.
- Competing hypothesis or confound: apparent lift may be only abstention from a harmful current repair, inherited from old `.25`, or caused by metric/source drift from v3r/v3z.
- Cheapest observation that separates them: exact A0D replay followed by paired safe shrink versus safe direction-oracle rendering on the same image/operator rows.

Mechanism sentence:

```text
For the exact v3z update128/heldout128 image-operator units, adding a privileged
bounded direction-line action family to privileged shrink/abstention should
increase safe rendered PSNR because spatial direction headroom exists outside
the current head's locally reachable update; no incremental lift would favor
action-space infeasibility or shrink-only abstention instead.
```

## Estimand And Risk Attribution

- Target population: exact v3z update128 and heldout128 train-derived images; heldout128 is the decision population and update128 is a transfer diagnostic.
- Analysis unit and grouping unit: clean-reference image, with D_ref/D_rep paired inside the image and never treated as independent samples.
- Intervention or factor contrast: safe augmented direction oracle versus safe shrink oracle, both selected by privileged clean-reference loss from fixed action grids.
- Reference/direct predecessor: current v3z Delta-u for failure attribution and old `.25` for deployable predecessor safety; old `.125` is the common safety anchor.
- Outcome, direction, and aggregation: higher paired PSNR is better; the primary effect is the minimum operator-specific mean lift, with image-group bootstrap uncertainty.
- Claim type: causal
- Identification assumptions and sensitivity limits: all actions share identical tensors/renderer and differ only by the fixed action set; clean targets make every selected action privileged and unusable for deployment.
- Minimum worthwhile effect or risk limit: worst-operator LCB95 at least `+0.005 dB` direction over shrink and worst-operator repairable-image-fraction LCB95 at least `0.20`.
- Equivalence/non-inferiority margin and independent source, if claimed: `+0.005 dB` comes from the pre-v4a v3r direction-repair SESOI; `20%` comes from the pre-v4a v3s wrong-direction repair floor.
- Common safety anchor: old `.125` for the same image/operator.
- Inherited-harm estimand: `max(MSE(old_.25)-MSE(old_.125),0)`.
- Candidate-total-harm estimand: `max(MSE(oracle_.25)-MSE(old_.125),0)`.
- Intervention-added-harm estimand: candidate total harm minus inherited harm, while the selected oracle additionally requires `MSE(oracle_.25) <= MSE(old_.25)+epsilon`.

## Design And Identifiability

- Design type: `feasibility_oracle`
- Why this is the cheapest design that identifies the estimand: the frozen model runs once per image/operator; all 65-point action grids are tensor-only and no training, policy fitting, or threshold search occurs.
- Experimental unit and randomization/pairing: fixed images with paired operators and paired action families; no randomized assignment is needed for deterministic oracle-set inclusion.
- Blocking, exclusion, failure, and missing-cell policy: no outcome row is excluded; missing names, nonfinite values, bound violations, replay mismatch, or absent action cells fail closed.
- Formal subgroup definitions and pre-intervention/independent source: split and operator only for the gate; A0D inherited-harm quartiles may be reported descriptively but cannot alter the action set or threshold.
- Primary comparison family and multiplicity treatment: one predeclared direction-over-shrink effect and one repairable-fraction guard; the bootstrap resamples images once per draw and takes the minimum across paired operators before its one-sided bound.
- Fractional-design resolution and alias structure, if applicable: not applicable.
- Negligible-interaction assumptions and targeted de-alias follow-up: no operator interaction is assumed negligible; the worst operator decides.
- Paired seeds/folds/data order/evaluation operators: seed `3407`, exact A0R order/folds, and D_ref/D_rep pairing are fixed.
- Natural groups and repeated grouped-split or leave-one-group-out plan: each filename prefix is one clean reference; the fixed historical population does not support a broader source-scene generalization claim.
- Split/seed uncertainty required for the claim: heldout128 is required for the decision; update128 cannot rescue heldout failure; no seed claim is made because the learned state is fixed.
- Uncertainty estimator and dependence/group structure: 4,000 PCG64(3407) paired image bootstrap draws; operator pairing is retained and the worst operator is computed within every draw.
- Sample/group/split/seed count justified by power or target interval width: all 128 heldout images are mandatory; the independent v3r effect was far above `+0.005 dB`, so failure to clear the fixed bound is decision-relevant for this population.
- Fixed-data attainable precision or smallest reliably detectable effect: the formal bound is the observed grouped-bootstrap precision; A1F is development screening and cannot establish final population safety.

## Adaptive Decision Paths

| Frozen trigger | Authorized next branch | Budget/data role | Stops or forbids |
| --- | --- | --- | --- |
| smoke exact replay and action integrity pass | full 128+128 A1F audit | `development_screening` | no training, policy, canary, or locked test |
| formal direction-over-shrink and repairable-fraction gates pass with all safety guards | new A1R representation-sufficiency design only | new development contract | no direct v4b training or confirmation |
| lift versus current exists but direction-over-shrink fails | terminal shrink/abstention diagnosis | development evidence only | no policy/router reopen and no representation training |
| safe direction headroom or structural validity fails | terminal learned-repair closeout or same-stage engineering repair, according to failure class | development or engineering only | A1R, v4b, v4c, canary, locked test |

- Evidence that branches may share: A0R final state, A0D rows, frozen assets, and A1F compact summaries.
- Evidence that must remain independent: any future representation selection and its confirmation data; locked test cannot select any action, representation, or threshold.
- Rule for an unlisted outcome: emit an R3 handoff and write a new amendment before any launch.

## Change

- Code branch: `codex/haze4k-v5-v4a-a1f-deltau-action-feasibility-20260714`.
- Exact code/config change: add a read-only audit that restores the v3z final head, verifies A0D rows, and evaluates fixed shrink and bounded direction-line action sets.
- Enabled mechanisms: exact final-state replay, privileged action-set inclusion, total/incremental risk decomposition, and paired grouped bootstrap.
- Explicitly disabled mechanisms: optimizer step, training, weight update, policy/selector fitting, threshold search, action-bound expansion, support expansion, canary, locked test, and checkpoint selection.
- Parameter/runtime/memory impact expected: zero deployable parameters; one frozen forward per image/operator plus tensor-only grid evaluation; compact outputs only.
- Initialization or no-op behavior: zero Delta-u exactly reproduces the old `.125/.25` predecessor; the restored current head must reproduce A0D.
- Resume policy: none; use a new output directory and run id after interruption.
- Defaults changed: no model or metric defaults; only the privileged fixed action sets are added.
- Defaults intentionally preserved: all v3z assets, data order, folds, operators, support, bounds, renderer, alpha values, and safety anchor.

## Preflight

| Check | Pass line | Result |
| --- | --- | --- |
| route identity | fresh local/cloud branch at the recorded A1F commit and clean cloud checkout | local complete; cloud required before launch |
| R3 authorization | exact v4a review tuple and SHA `cddb543c67c5c1e167686bb3bed08dc25ca2c504a0835e899b123973e6b62a99` authorizes A1F design/implementation only | complete |
| final state | unique A0R r1 epoch16/update512 final state SHA matches `ed0832f...` | source probe complete; runtime recheck required |
| A0D alignment | raw table has 512 rows and SHA `045fecdb6701b0d7cad06772e15d7d2f2b5330db075f8c57cf9043d101b0dd05` | source identity complete; exact replay pending S0 |
| action contract | 65-point grids, old support, v3j bounds, and deterministic tie rules are exact | static complete; tensor checks pending S0 |
| data protection | only train-derived update128/heldout128 and `locked_test_touched=false` | static complete; dynamic recheck required |
| GPU capacity | selected device has at least 18,000 MiB free and at most 10% utilization before launch | dynamic check required |

## Mechanism Metrics

| Metric | Why it matches the route | Gate subset | Final artifact |
| --- | --- | --- | --- |
| direction-over-shrink worst-operator LCB95 | isolates direction headroom from abstention | heldout128 | `v4a_a1f_bootstrap_summary.json` |
| repairable-image fraction LCB95 | requires broad rather than one-image oracle value | heldout128 | `v4a_a1f_bootstrap_summary.json` |
| exact A0D replay max discrepancy | excludes metric/source drift | smoke and formal | `v4a_a1f_closeout.json` |
| total and intervention-added harm | preserves common-anchor safety attribution | both splits/operators | `v4a_a1f_operator_summary.csv` |
| action-grid selection and bound saturation | verifies the privileged mechanism acted inside the fixed space | both splits/operators | `v4a_a1f_operator_summary.csv` |

An image/operator is repairable only when the selected direction-line action
has rendered `.25` MSE lower than the selected shrink action by more than the
frozen numerical tolerance. A tie or shrink fallback is not repairable.

## Controls

| Control | Purpose | Pass line |
| --- | --- | --- |
| A0D current-head replay | validate the exact failed state and renderer | MSE max difference `<=1e-12`, PSNR max difference `<=1e-9 dB` |
| zero Delta-u | validate predecessor/no-op semantics | exact old `.125/.25` tensor equality |
| privileged shrink oracle | separate abstention/amplitude from direction | complete 65-point grid including zero and current |
| fixed direction target | align to v3r geometry without post-result action search | `support*clip(4*(J-base),-B,B)` and bounded Delta-u difference |
| paired operators | prevent one operator from hiding failure | both present for every image and worst operator decides |

## Evidence-Role Ledger

| Evidence source or groups | Role | Allowed uses | Forbidden uses |
| --- | --- | --- | --- |
| 8 update + 8 heldout smoke names | `engineering_debug` | identity, replay, finite/bound/grid checks | threshold or action selection |
| full update128 | `development_screening` | transfer diagnostic only | rescue heldout gate or confirmation |
| full heldout128 | `development_screening` | A1F feasibility decision | deployment, confirmation, or later candidate proof |
| Haze4K locked test | `sealed_final` | none in A1F | all access |

- Candidate/threshold/operator freeze point: all action sets, thresholds, operators, and tie rules freeze in this card before smoke.
- Independent confirmation contract: none in A1F; a pass authorizes only new representation-sufficiency design using a separate selection/confirmation plan.
- Nested group-respecting resampling contract, if no separate confirmation set: not used for a promotion claim.
- Final sealed-use authorization and one-use policy: not authorized.
- Post-sealed rule (`report/close only; no tuning or reselection`): not applicable because sealed data remain untouched.

## Fair Run Contract

- Training or inference budget: smoke uses the first 8 names per split; formal uses all 128 per split; each image/operator gets exactly one frozen head forward and both fixed 65-point grids.
- Batch/sample policy: one image at a time; candidate tensors may be chunked only for memory and must yield identical float32 renders.
- Optimizer: none.
- Schedule: smoke then formal only after typed smoke pass.
- Loss weights: none; privileged selection minimizes rendered `.25` MSE subject to the frozen safety constraints.
- Random seed policy: deterministic seed `3407`; only bootstrap resampling uses randomness.
- Evaluation cadence: one terminal summary per stage.
- Checkpoint cadence: none.
- Hardware/runtime assumptions: `convir-4090`, explicit cu121 Python, one CUDA GPU selected by dynamic preflight.
- Allowed resume behavior: none.
- Sample-size policy: no exclusions or substitutions; formal requires exactly 256 images and 512 paired image/operator rows.
- Dependency/version assumptions: v4a/v3z/v3s/v3p commits and all asset hashes must match the source manifest.
- Selected decision profile: `feasibility_oracle`
- Learned-state retention required: no: the route restores one immutable learned state and creates no learned state.
- Omitted or specialized stage rationale: no training, policy replay, confirmation, canary, or locked-test stage can answer the feasibility question.

## Gates

| Stage | Estimand/question | Evidence role and budget/sample scope | Gate type, threshold source, and multiplicity rule | `PASS` authorizes |
| --- | --- | --- | --- | --- |
| S0 alignment smoke | Does the runner reproduce A0D and obey the frozen action/bound/data contract? | `engineering_debug`, first 8 names from each split and both operators | structural: exact identities, all rows finite, MSE `<=1e-12`, PSNR `<=1e-9 dB`, no bound/support violation | A1F formal only |
| A1F formal | Does bounded direction add safe headroom beyond privileged shrink? | `development_screening`, all update128/heldout128; heldout decides | scientific utility plus safety: 4,000 paired image bootstrap draws; worst-operator direction-over-shrink LCB95 `>=+0.005 dB`, repairable-fraction LCB95 `>=0.20`, all selected actions anchor/predecessor non-worse, and zero severe/hard regression versus old `.25` | R3 handoff for separate A1R design only |
| independent confirmation | not part of A1F | none | prohibited in this route | none |
| sealed final | not part of A1F | none | prohibited in this route | none |

## Analysis Plan

- Per-sample or subgroup analysis: retain cloud-only rows by split/operator and compact aggregate summaries; inherited-risk quartiles are descriptive only.
- Visual or qualitative analysis: none required for the feasibility gate.
- Complexity analysis: record wall time and peak GPU memory only; no deployable cost claim.
- Robustness or held-out analysis: update128 and heldout128 are reported separately; only heldout128 decides and both operators must pass.
- Regression analysis: none.
- Main-effect/interaction and alias analysis, if applicable: not factorial; report paired operator effects and their worst-case bootstrap.
- Group/split/seed uncertainty and sensitivity analysis: paired image bootstrap; no seed or population claim beyond the fixed final state.
- Screening-selection versus confirmation analysis: every A1F result is privileged development screening and cannot confirm a representation.
- Required docs to update: this card, evidence README, typed closeout, CHD-RM index, and global index at terminal handoff.
- Required artifacts to retain: runner, source manifest, closeout, operator summary, bootstrap summary, hashes, and compact status.
- Required artifacts to delete or keep external: per-image/action rows, tensors, images, logs, and any cached frozen sample remain in cloud `RUN_ROOT`.
- Evidence package contents: route card, README, closeout JSON, source manifest JSON, operator summary CSV, and bootstrap summary JSON.
- Evidence package audit: explicit text paths only; no raw per-image/action table, checkpoint, tensor, image, or log enters Git.

## Results And R3 Review

- The repaired S0 smoke passed 32/32 image/operator rows with exact A0D MSE
  and PSNR replay, zero-action tensor discrepancy, bound excess, and support
  excess all `0.0`.
- Formal completed 512/512 rows and 4,000 PCG64(3407) paired image bootstrap
  draws with operator pairing and the worst operator retained within each draw.
- Heldout worst-operator direction-over-shrink was `+0.128716 dB` with LCB95
  `+0.105475 dB`, above the preregistered `+0.005 dB` gate.
- Heldout repairable-image fraction was `0.765625` with LCB95 `0.6953125`,
  above the preregistered `0.20` gate.
- Heldout worst-operator lift versus old `.25` was `+0.224309 dB` with LCB95
  `+0.200613 dB`; every selected row was anchor- and predecessor-nonworse,
  with zero severe/hard regression.
- The safe direction oracle also improved over the current v3z repair by
  worst-operator `+0.199285 dB`, LCB95 `+0.158832 dB`.
- The result identifies safe bounded direction headroom beyond privileged
  shrink/abstention at the exact failed v3z state. It favors representation or
  reachable-action insufficiency over action-space infeasibility.
- This remains privileged `development_screening`: clean targets select each
  action, so it is nondeployable and cannot authorize candidate training,
  promotion, canary, or locked-test access.

## Decision

- Decision label: `V4A_A1F_SAFE_DIRECTION_HEADROOM_PASS_AUTHORIZE_A1R_REPRESENTATION_SUFFICIENCY_DESIGN_ONLY`.
- Image/global metric reason: the heldout128 safety failure must be tested at the same image/operator level without allowing update128 to rescue it.
- Mechanism reason: direction-over-shrink isolates action direction from optimizer, window, and abstention explanations.
- Preservation or regression reason: every selected action must be non-worse than old `.125` at `.125` and old `.25` at `.25` before aggregate lift is considered.
- Inherited harm versus anchor: old `.25` burden is retained explicitly.
- Candidate total harm versus anchor: selected-oracle burden uses the same old `.125` anchor.
- Intervention-added harm versus predecessor: selected `.25` must not exceed old `.25` beyond numerical tolerance.
- Group/split/seed uncertainty and interaction reason: image-group bootstrap retains paired operators and uses the worst operator per draw.
- Evidence role and independence reason: privileged development evidence can decide whether A1R is informative but cannot select a deployable candidate.
- Cost/deployability reason: no training or deployable change; privileged GT use makes A1F nondeployable by design.
- What this decides next: a fresh, preregistered A1R representation-sufficiency
  audit only; the A1F oracle action and heldout outcomes cannot train or select
  that representation.
- Typed closeout path: `RUN_ROOT/v4a_a1f_formal_r1/v4a_a1f_closeout.json`.
- `PASS` authorizes: `A1R_ROUTE_DESIGN_ONLY`.
- `INCONCLUSIVE` authorizes: `SAME_STAGE_ENGINEERING_OR_PREDECLARED_EVIDENCE_REPAIR_ONLY`.
- `FAIL` stops: `A1R_V4B_V4C_POLICY_CANARY_AND_LOCKED_TEST`.
