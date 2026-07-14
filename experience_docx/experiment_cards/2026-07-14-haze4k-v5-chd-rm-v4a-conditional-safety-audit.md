# Haze4K v5 CHD-RM v4a Conditional Safety Audit

Date: 2026-07-14

Status: `A0P_COMPLETE_R3_AUTHORIZE_A1F_FEASIBILITY_DESIGN_ONLY`

## Scope

- Project: ConvIR-B Haze4K CHD-RM v5.
- Model family: frozen ConvIR-B plus the historical v3z output-side Delta-u repair head for reconstruction and diagnostics only.
- Dataset or task: Haze4K train-derived internal data; no canary or locked Haze4K test access.
- Primary objective: identify whether the v3z heldout safety failure is attributable to reconstruction/metric, risk-stratum, update-contract, action-space, or representation limitations before any new repair training.
- Main metric: canonical per-image paired rendered safety and utility contrasts after clamp, with exact v3z aggregate reconstruction as the first integrity gate.
- Secondary metrics: anchor/total/added harm, relative harm, CVaR10, p05 paired PSNR delta, severe/hard counts, support-normalized burden, actual-step constraint change, and QP feasibility.
- Execution environment: local WSL for editing and static checks only; runtime only on `convir-4090`.
- GitHub rules commit: `7df7ae151911e8de077849215fdec8383c0b4dba`.
- Local WSL path, if used for editing/static checks: `/home/ubuntu/workspace/ConvIR-B-v4a-conditional-safety-audit-20260714`.
- GitHub route branch and source commit: `codex/haze4k-v5-v4a-conditional-safety-audit-20260714` from `7df7ae151911e8de077849215fdec8383c0b4dba`; exact v3z replay parent `3caddcc5265732e5be77e3404119a28cb28c11e6`.
- Cloud `REMOTE_REPO`: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v4a-conditional-safety-audit-20260714`.
- Cloud `RUN_ROOT`: `/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v4a_conditional_safety_audit_20260714`.
- Cloud `EVID_STAGE`: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v4a-conditional-safety-audit-20260714/experience_docx/experiment_logs/haze4k_v5_chd_rm_v4a_conditional_safety_audit_20260714`.
- Explicit cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.

## Agent Execution Routing

Use `MODEL_AGENT_COST_ROUTING_PROTOCOL.md` for role qualifications and dispatch mechanics.

| Applicable scope | Task class | Minimum role | Boundary action |
| --- | --- | --- | --- |
| scientific design / gate contract | `R3_SCIENTIFIC_AUTHORITY` | `frontier` | `dispatch=not_amortized: current route design and first diagnostic remain one R3 envelope` |
| workspace / runner engineering | `R2_ENGINEERING_CONTROL` | `balanced` | `dispatch=not_amortized: runner implements the frozen A0R-passed A0P contract` |
| preflight / launch / monitor / evidence fetch | `R1_BOUNDED_EXECUTION` | `fast` | `standalone_repetition after a machine-verified typed authorization` |
| result interpretation / terminal verdict | `R3_SCIENTIFIC_AUTHORITY` | `frontier` | `required_escalation before any branch or verdict decision` |
| unchanged-verdict archival / sync | `R2_ENGINEERING_CONTROL` | `balanced` | `major_handoff after a reviewed closeout` |

## Baseline Contract

- Baseline implementation: v3z projected repair runner at replay commit `3caddcc5265732e5be77e3404119a28cb28c11e6`.
- Baseline checkpoint or initialization: official `haze4k-base.pkl`, SHA-256 `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`, and v3z zero-init repair head.
- Evaluation entrypoint: v3z `candidate_predictions` and `candidate_metrics` under the canonical `.125/.25` rendered clamp contract.
- Training entrypoint: v3z projected 16-epoch repair loop, eight render-warmup epochs followed by eight projected-safety epochs.
- Dataset and split: exact v3z first 128 `v3j_controller_train` names update the head; the next disjoint 128 names are held out; names and fold mapping must hash-match the v3z manifest.
- Preprocessing and decoding: exact v3z Haze4K train loader, padding, crop, support construction, frozen operator cache, and clamp/decode path.
- Metric implementation: original rendered MSE for exact reconstruction; v4a additionally records canonical per-image MSE/PSNR and risk decomposition from the same tensors.
- Reproduced baseline result: historical heldout rendered-MSE reduction `0.013561150457617625`, heldout anchor `1.2388485188807863e-06`, and heldout harm `3.6074538840580317e-06`.
- Known reproduction gap: original cloud run retained aggregate history but not repair checkpoints, optimizer states, RNG states, per-image rows, or per-update traces.
- Reference entrypoints that must remain stable: `chd_rm_v3x_projected_safety_constraint.py`, its v3w/v3s imports, frozen operator manifests, and the exact asset paths recorded in the v3z source manifest.
- Checkpoint/export/resume contract: A0R never resumes an incomplete run; each replicate starts from the exact zero-init state and writes new immutable runtime outputs only under this route's `RUN_ROOT`.

## Most Valuable Attempt

- Why this is the highest-value next attempt: it resolves whether the apparent conditional safety failure is scientifically interpretable before paying for a new representation or training route.
- Target failure or opportunity: v3z utility remained positive across a disjoint 128-image slice while aggregate anchor and harm exceeded unsafe diagnostic references.
- Cheap preflight evidence: exact source/asset identity audit and two instrumented deterministic reconstructions of the closed v3z contract.
- Earliest decisive gate: A0R must reproduce all original aggregate checkpoints and the two new reconstructions must agree within the written numerical contract.
- Expected cost or attempt-count saving: a single roughly two-times-v3z reconstruction prevents speculative optimizer, latent, basis, and policy runs from sharing an unidentified failure.
- What success decides: authorizes per-image risk tomography and paired update-mechanism auditing only.
- What failure decides: closes causal interpretation of historical v3z numerical behavior and blocks all A0P/A0M/A1/v4b continuation pending a separate reproduction repair route.
- Why a cheaper diagnostic is not enough: no retained v3z learned state exists, so the required per-image and actual-step evidence cannot be recovered by reading the historical closeout.

## Hypothesis

- Observed failure: the fixed v3z projected head improved heldout rendered MSE but failed aggregate heldout anchor/harm references.
- Target mechanism: the historical sequential raw-gradient projection can be locally feasible while AdamW's preconditioned, decayed parameter displacement and fixed small risk windows fail to control conditional safety.
- Null hypothesis: exact reconstruction shows no material update-contract discrepancy; the observed safety failure is fully explained by the pre-existing risk distribution and/or the chosen aggregate reference.
- Preferred causal hypothesis: holding a saved state and risk window fixed, proposal-space constrained updates reduce actual post-step safety violations relative to the historical update contract without reducing matched local utility.
- Competing hypothesis or confound: heldout risk is driven by unobserved image/operator state that cannot be controlled by optimizer correction; a reference/aggregation mismatch can also create an apparent transfer failure.
- Cheapest observation that separates them: paired same-state shadow updates plus per-image inherited/total/added-risk rows from an exact reconstruction.

Mechanism sentence:

```text
For the fixed v3z image/operator unit, a proposal-space constrained update
relative to sequential raw-gradient projection should lower actual post-step
safety violation because it constrains the parameter displacement that AdamW
applies; no paired improvement favors risk-stratum, metric, or representation
explanations instead.
```

## Estimand And Risk Attribution

- Target population: the exact v3z 128 update images and disjoint 128 heldout images for A0; later train-derived data are reserved and are not consumed by A0R.
- Analysis unit and grouping unit: one clean-reference image with `D_ref` and `D_rep` paired inside the image; each historical filename prefix is a unique clean-reference group, so inference is image-level rather than scene-cluster generalization.
- Intervention or factor contrast: historical sequential projected gradient plus AdamW versus predeclared same-state update methods in A0P; A0R itself changes only observability, not update mathematics.
- Reference/direct predecessor: old `.125` is the common safety anchor, old `.25` is the direct predecessor for repair utility, and the frozen historical v3z trajectory is the numerical reference.
- Outcome, direction, and aggregation: paired per-image/operator rendered MSE and PSNR after clamp; report worst-operator mean/LCB for utility and worst-group/UCB for safety without averaging an operator failure away.
- Claim type: associational
- Identification assumptions and sensitivity limits: A0D is descriptive because image strata are not randomized; A0P can identify local same-state method effects only, not end-to-end training generalization; each clean reference occurs once.
- Minimum worthwhile effect or risk limit: A0R requires numerical equivalence; A0P is informative only if a method changes actual held-risk-bank violation without a nontrivial matched utility loss; no candidate-quality promotion claim is allowed in v4a.
- Equivalence/non-inferiority margin and independent source: A0R requires every historical aggregate field to differ by at most `max(1e-9, five times the absolute R1-R2 difference)` and exact zero for no-op fields; this is a numerical integrity tolerance, not a safety threshold.
- Common safety anchor: the exact old `.125` rendered prediction for the same image/operator and label.
- Inherited-harm estimand: `max(MSE(old_.25)-MSE(old_.125), 0)` per image/operator.
- Candidate-total-harm estimand: `max(MSE(new_.25)-MSE(old_.125), 0)` per image/operator.
- Intervention-added-harm estimand: `candidate_total_harm-inherited_harm`, reported together with `max(MSE(new_.25)-MSE(old_.25), 0)` to expose sign and positive burden.

## Design And Identifiability

- Design type: `hybrid: exact reconstruction, paired factorial same-state audit, and preregistered adaptive branches`.
- Why this is the cheapest design that identifies the estimand: exact reconstruction establishes trustworthy states; the 3-by-3 same-state screen isolates update method and window estimator before a new learned representation is considered.
- Experimental unit and randomization/pairing: A0R uses fixed historical order; A0P blocks by retained pre-update state and pairs all update methods on the same state/window, with window assignment generated from a stored seed.
- Blocking, exclusion, failure, and missing-cell policy: no image/operator is excluded for outcome; an unavailable asset, nonfinite tensor, mismatched hash, QP failure, or missing factor cell is retained and causes the relevant integrity or feasibility gate to fail.
- Formal subgroup definitions and pre-intervention/independent source: operator, filename haze parameters, old-step energy, support fraction, old `.25` inherited harm, and clamp fraction are computed from frozen pre-repair values; post-result buckets are exploratory only.
- Primary comparison family and multiplicity treatment: A0P primary family is actual post-step safety change across the three methods and three window estimators; use a max-statistic paired bootstrap across the full family, and retain all unadjusted values as descriptive.

| Factor | Levels | Main effect required? | Required interactions |
| --- | --- | --- | --- |
| update method | historical sequential gradient, exact gradient-intersection, actual-proposal projection plus backtracking | yes | update method by window estimator |
| risk window estimator | historical fixed4, shuffled16, pre-stratified32 | yes | update method by window estimator |

- Fractional-design resolution and alias structure: not applicable: A0P is a complete 3-by-3 paired screen at retained states, not a fractional factorial.
- Negligible-interaction assumptions and targeted de-alias follow-up: no interaction is assumed negligible; the two-factor interaction is estimated, while end-to-end trajectory effects require the separately authorized A0M branch.
- Paired seeds/folds/data order/evaluation operators: A0R uses seed `3407` and historical fold mapping exactly; A0P preserves state/window/operator pairing; any A0M branch pairs seeds, initialization, fold assignment, and data order across cells.
- Natural groups and repeated grouped-split or leave-one-group-out plan: A0 is limited to the fixed historical population; later A1/v4b must use repeated stratified image splits and disclose that filename groups are unique unless stronger source-scene metadata is recovered.
- Split/seed uncertainty required for the claim: A0R requires two independent process reconstructions; A0P uses paired state-level bootstrap and makes no population promotion claim; A0M/A1/v4b require a frozen multi-seed plan before launch.
- Uncertainty estimator and dependence/group structure: paired cluster bootstrap resamples images while retaining both operator rows; A0P additionally blocks by pre-update state; ties and numerical gray-zone rows remain explicit.
- Sample/group/split/seed count justified by power or target interval width: A0R uses every 128+128 historical row because exact reconstruction is the estimand; A0P uses every retained safety update and a predeclared 1,000-resample max-statistic bootstrap; later formal tail gates require at least 30 tail images per operator/group.
- Fixed-data attainable precision or smallest reliably detectable effect: A0 has 256 image/operator pairs per split and is diagnostic only; its intervals cannot establish low-haze population preservation or final utility non-inferiority.

## Adaptive Decision Paths

| Frozen trigger | Authorized next branch | Budget/data role | Stops or forbids |
| --- | --- | --- | --- |
| A0R reproduces original trace and emits complete state manifest | A0D and A0P | same 256 rows, `development_screening` | no model training, policy, canary, or locked test |
| A0R numerical-equivalence fails | separate reproduction-repair route only | `engineering_debug` | A0D causal interpretation, A0P, A0M, A1, v4b, v4c, and locked test |
| A0P paired proposal-space method improves actual safety with no material local utility loss | A0M matched end-to-end current-head screen | new development-only contract | no internal-latent model or policy yet |
| A0P finds no local feasible correction | A1F action-space feasibility oracle | development-only privileged analysis | no optimizer retuning or policy route |
| A0M corrected current head passes its later written development gate | separate current-head confirmation card | untouched route-specific confirmation | no architecture expansion before confirmation |
| A0M fails and A1F shows bounded action headroom | A1R representation sufficiency audit | nested `development_screening` | no direct v4b training or policy |
| A1F lacks safe bounded headroom or A1R fails replay and controls | terminal learned-repair closeout | development evidence only | v4b, v4c, continuous alpha, canary, and locked test |

- Evidence that branches may share: v3z historical assets, A0R retained states, and A0 development rows may be reused for diagnostic comparisons only.
- Evidence that must remain independent: future route-specific confirmation rows and the locked Haze4K test cannot select update method, window, basis, representation, threshold, checkpoint, policy, or fallback.
- Rule for an unlisted outcome: write a new route card amendment and do not launch an unlisted branch.

## Change

- Code branch: `codex/haze4k-v5-v4a-conditional-safety-audit-20260714`.
- Exact code/config change: add an instrumented wrapper around the immutable v3z replay source that retains model/optimizer/RNG/sampler state and writes canonical per-image and projection traces without changing the v3z forward, loss, update order, or hyperparameters.
- Enabled mechanisms: exact replay observability, per-image/operator risk decomposition, state-level paired update audit, and cloud-only learned-state retention.
- Explicitly disabled mechanisms: model architecture changes, backbone unfreeze, new repair training, policy/confidence training, continuous alpha, canary, locked test, reference relaxation, and v3z hyperparameter tuning.
- Parameter/runtime/memory impact expected: A0R preserves the 2,883-parameter head and 16-epoch training; runtime storage increases by retained small-head states and traces under `RUN_ROOT`, with no inference-path change.
- Initialization or no-op behavior: exact v3z zero-init and S0 no-op contract are preserved; any no-op discrepancy is an integrity failure.
- Resume policy: no scientific resume; an interrupted replicate may rerun only from a new output directory with the same frozen contract and an infrastructure closeout.
- Defaults changed: no v3z numerical default changes; v4a adds write-only instrumentation outside the optimization semantics.
- Defaults intentionally preserved: seed, names, folds, operators, bounds, warmup, epochs, risk-window, AdamW settings, gradient clipping, rendered metric, and original support/clamp logic.

## Preflight

| Check | Pass line | Result |
| --- | --- | --- |
| clean route identity | local branch HEAD and pushed cloud checkout equal the recorded v4a commit | required before dynamic launch |
| immutable replay source | fresh cloud v3z source snapshot is clean and exactly `3caddcc5265732e5be77e3404119a28cb28c11e6` | required before dynamic launch |
| asset identity | every v3z manifest asset path exists and SHA-256 matches its recorded value | required before dynamic launch |
| static runner contract | v4a runner parses, imports the frozen source, and does not alter v3z math paths | required before `PLANNED` closeout |
| output isolation | all new checkpoints, traces, logs, and tables resolve under v4a `RUN_ROOT` and no historical output is overwritten | required before dynamic launch |
| locked-test exclusion | command contains only Haze4K train-derived source split and reports `locked_test_touched=false` | required before dynamic launch |

## Mechanism Metrics

| Metric | Why it matches the route | Gate subset | Final artifact |
| --- | --- | --- |
| original-versus-R1/R2 aggregate discrepancy | establishes whether state reconstruction is trustworthy | A0R 128 update plus 128 heldout | `v4a_a0r_reproduction_summary.json` |
| per-image inherited, total, and added harm | separates predecessor burden from repair-added risk | A0D both operators | `v4a_a0d_group_tail_summary.csv` |
| actual parameter displacement constraint change | tests the AdamW-gradient mismatch directly | A0P retained safety states | `v4a_a0p_step_summary.csv` |
| actual post-render safety and utility | rejects purely first-order safety claims | A0P independent risk bank | `v4a_a0p_poststep_summary.csv` |
| retained-state completeness | makes later trajectory claims auditable | A0R all safety updates | `v4a_a0r_trace_manifest.json` |

## Controls

| Control | Purpose | Pass line |
| --- | --- | --- |
| two independent exact reconstructions | measures implementation/numerical reproducibility rather than trusting one rerun | both traces meet the A0R numerical contract |
| historical sequential update | anchors every A0P factor cell to the closed v3z method | same-state historical cell is present for every retained state/window |
| no-op S0 replay | detects accidental initialization or clamp drift | max prediction and Delta-u difference remain exactly zero |
| shuffled and stratified risk windows | distinguishes fixed-window estimator sensitivity from update-method effect | stored window seeds and names reproduce each assigned cell |
| full actual render after shadow step | rejects a first-order-only safety claim | post-clamp metrics are evaluated for every completed factor cell |

## Evidence-Role Ledger

| Evidence source or groups | Role | Allowed uses | Forbidden uses |
| --- | --- | --- | --- |
| v3z original 128 update and 128 heldout names | `engineering_debug` | hash, source, and numerical reconstruction checks | confirmation, threshold fitting, promotion |
| v3z reconstructed per-image rows and retained states | `development_screening` | A0D/A0P factor comparison and adaptive branch selection | independent confirmation, policy selection, sealed claims |
| future precommitted train-derived confirmation partition | `confirmation` | frozen repair/operator/policy inference in a later route card | A0/A1 factor, threshold, or checkpoint selection |
| Haze4K locked test | `sealed_final` | one final fully frozen system evaluation in a later authorized route | any debugging, selection, tuning, or rerun |

- Candidate/threshold/operator freeze point: no candidate exists in v4a; any A0M/A1/v4b candidate must freeze all code, assets, bounds, basis, threshold, operator, and fallback before confirmation.
- Independent confirmation contract: not applicable: v4a is diagnostic and cannot support a candidate confirmation claim; later route cards must define a new untouched confirmation contract.
- Nested group-respecting resampling contract: A0P resamples image clusters within retained states; later A1/v4b must nest all basis/representation/threshold selection inside outer training partitions.
- Final sealed-use authorization and one-use policy: no sealed use is authorized by v4a; a later typed confirmation closeout must explicitly authorize the sole locked-test command.
- Post-sealed rule: report/close only; no tuning, reselection, new threshold, or continuation from sealed evidence.

## Fair Run Contract

- Training or inference budget: A0R runs two independent exact 16-epoch replays of 128 update images plus 128 heldout evaluations; A0P is a saved-state shadow audit and does not train a candidate.
- Batch/sample policy: original deterministic fixed4 windows for A0R; A0P compares fixed4, shuffled16, and pre-stratified32 with stored assignments.
- Optimizer: original AdamW with learning rate `0.0005`, weight decay `0.00001`, and gradient clip norm `0.1` for A0R.
- Schedule: original eight render-warmup epochs followed by eight projected-safety epochs.
- Loss weights: original rendered MSE warmup and original v3z projected anchor/harm/margin/CVaR constraints; instrumentation never inserts a loss term.
- Random seed policy: A0R replicates independently reset Python, NumPy, CPU Torch, and CUDA Torch states to `3407`; A0P uses recorded state and window seeds.
- Evaluation cadence: initial, warmup epoch 8, final epoch 16, every retained safety update trace, and per-image/operator canonical evaluation at each retained evaluation point.
- Checkpoint cadence: before and after every safety update, plus initial, warmup, and final states; raw state stays in `RUN_ROOT`.
- Hardware/runtime assumptions: one `convir-4090` GPU, explicit cu121 Python, and no concurrent route writes to the v4a output root.
- Allowed resume behavior: exact rerun only after `FAILED_INFRA` or `FAILED_COMMAND`; no result-dependent restart and no reuse of an existing output directory.
- Sample-size policy: all historical 128 update and 128 heldout rows are used; no subset result authorizes a model candidate.
- Dependency/version assumptions: v3z source commit, v3s/v3p commits, official checkpoint, operator artifacts, and Python package identity must be manifest-hashed before launch.
- Selected decision profile: `hybrid: audit_evaluation followed by factorial_screening only after A0R integrity passes`.
- Learned-state retention required: yes
- Omitted or specialized stage rationale: no training, policy replay, canary, confirmation, or locked-test stage is included because v4a only identifies which later route is scientifically justified.

## Gates

| Stage | Estimand/question | Evidence role and budget/sample scope | Gate type, threshold source, and multiplicity rule | `PASS` authorizes |
| --- | --- | --- | --- | --- |
| A0R | Can the original v3z numerical contract be reconstructed and retained? | `engineering_debug`, two exact 16-epoch 128+128 replays | numerical-equivalence: all original aggregates meet the written tolerance, no-op fields are exact, all trace states are manifested; every field is required | A0D and A0P only |
| A0D | Where do inherited, total, and added risks concentrate under the reconstructed contract? | `development_screening`, all per-image/operator rows | information: canonical schema, hashes, and all predeclared group/tail summaries exist; descriptive only | A0P interpretation only |
| A0P | Does method/window choice improve actual same-state safety without a matched local utility loss? | `development_screening`, all retained safety states and 3-by-3 paired cells | selection: max-statistic paired bootstrap across the complete family; missing/infeasible cells remain failures | A0M or A1F only by frozen branch trigger |
| A0M | Does the selected update-contract repair improve an end-to-end current-head development trajectory? | `development_screening`, later predeclared paired multi-seed screen | scientific utility and safety: route amendment fixes margins, seeds, and outer evaluation before launch | separate confirmation card only |
| A1F/A1R | Does a deployable bounded action space and representation retain safe oracle headroom? | `development_screening`, later nested folds | feasibility/information: privileged action oracle then replay/control gate | separate v4b card only |
| locked test | Is the fully frozen system final-ready? | `sealed_final`, no v4a budget | prohibited: no v4a closeout can authorize this stage | none |

## Analysis Plan

- Per-sample or subgroup analysis: retain cloud-only per-image/operator rows at initial, warmup, and final; summarize predeclared operator, haze, support, step-energy, inherited-risk, and clamp-risk groups.
- Visual or qualitative analysis: inspect only a fixed worst-burden/worst-utility contact set after A0D summaries; images are diagnostic and remain cloud-only.
- Complexity analysis: record state count, runtime, peak memory, and trace storage; no inference-cost claim is made in v4a.
- Robustness or held-out analysis: original heldout128 is diagnostic, not confirmation; A0P evaluates every shadow cell on its independent held-risk bank.
- Regression analysis: fit descriptive pre-intervention covariate associations with added harm; do not use them to define post-hoc formal subgroups.
- Main-effect/interaction and alias analysis: estimate update-method, window-estimator, and their paired interaction; do not infer full-trajectory causality from one-step effects.
- Group/split/seed uncertainty and sensitivity analysis: image-cluster bootstrap with paired operators; report the effect of risk-window seeds and retain all assigned windows.
- Screening-selection versus confirmation analysis: v4a has only engineering/debug and development evidence; any later selected method must receive a fresh confirmation contract.
- Required docs to update: this card, v4a evidence README, CHD-RM index, and global experiment index at a major or terminal handoff.
- Required artifacts to retain: runner, manifests, aggregate summaries, typed closeout, state manifest, state hashes, window assignments, and compact trace summary.
- Required artifacts to delete or keep external: checkpoints, optimizer/RNG states, raw per-image rows, gradients, arrays, images, and full logs remain under `RUN_ROOT`.
- Evidence package contents: README, source manifest, reconstruction summary, group/tail summary, projection summary, retained-state manifest summary, typed closeout, and all relevant SHA-256 values.

## Learned-State Retention

- Retained steps/epochs/factor cells and why each is needed: initial, post-warmup, final, and pre/post every epoch 9-16 update are retained to reproduce state-level A0P comparisons; every 3-by-3 A0P cell records its source state and window identity.
- Model/checkpoint state path and hash contract: `RUN_ROOT/a0r/replicate-id/states/*.pt` contains only the repair head state with SHA-256 recorded in `trace_manifest.json`.
- Optimizer/scheduler state contract: every retained state records AdamW state; no scheduler exists and the manifest explicitly records `scheduler=null`.
- RNG states required and unavailable-state disclosure: Python, NumPy, CPU Torch, CUDA Torch-all-device states are retained; dataloader workers are not used and their unavailable-state set is empty.
- Data-order/sampler identity: exact names, folds, update index, risk window, window seed, and full per-epoch order are retained in JSON/CSV manifests.
- Config hash, code commit, Python/environment identity, and parent checkpoint: every trace row records v4a commit, v3z/v3s/v3p commits, config SHA-256, `PY --version`, Torch/CUDA versions, and official checkpoint SHA-256.
- Trace-manifest path and schema: `RUN_ROOT/a0r/trace_manifest.json` has versioned state, update, asset, and hash rows; schema version is recorded in its top-level object.
- Cloud retention/deletion policy: retain raw state through the terminal v4a decision and one subsequent verification window; delete only after compact evidence is synced and the decision card records deletion approval.
- Compact GitHub evidence: only paths, hashes, counts, schema version, aggregate summaries, and typed closeout are committed; no state, raw per-image table, or gradient tensor enters Git.

## A0D/A0P Amendment

This amendment was frozen after the A0R gate passed. It is the only authority
for A0D/A0P method mathematics, windows, endpoints, bootstrap, and typed
triggers. It does not alter the A0R numerical contract or authorize A0M, A1,
v4b, v4c, canary, locked test, promotion, or checkpoint selection.

### A0R Prerequisite

- Canonical state source: A0R `r1` only, with every `pre` state from epochs
  9-16. `r2` is a reconstruction control and cannot select a method, window,
  parameter, or threshold.
- Required typed tuple: `state=COMPLETED_GATE_PASS`,
  `decision=V4A_A0R_REPRODUCTION_PASS_AUTHORIZE_A0D_AND_A0P`, and
  `authorizes=A0D_AND_A0P_ONLY`.
- Required numerical facts: no-op exact; all historical/R1, historical/R2,
  and R1/R2 aggregate and history discrepancies are zero under tolerance
  `1e-9`; each replicate has 515 states, 512 projection rows, and 1,280
  per-image rows.
- `D_ref` and `D_rep` remain paired frozen evaluation environments inside every
  image cluster. They are not update-method levels and cannot be averaged as
  independent evidence.

### A0D Contract

- Population: report A0R `r1` final rows separately for `update128` and
  `heldout128`, with both frozen operators and no exclusions or substitutions.
- Per image/operator quantities, using the canonical rendered MSE `e`, are
  `H_inherited=max(e_old_250-e_old_125,0)`,
  `H_total=max(e_new_250-e_old_125,0)`,
  `H_intervention=H_total-H_inherited`, and
  `H_predecessor_positive=max(e_new_250-e_old_250,0)`. The last quantity must
  never be relabeled as total intervention harm.
- For each operator and split, report mean, median, p95, CVaR10, PSNR mean and
  p05, severe count (`delta_psnr <= -0.2 dB`), hard count
  (`delta_psnr <= -0.5 dB`), and burden divided by support fraction. A
  zero-support row is retained and marks the normalized-burden field invalid.
- Freeze quartile groups from A0R initial pre-intervention values only, using
  linear 25/50/75 percentiles and name-order tie breaking, for filename haze
  parameters, old-step energy, support fraction, inherited harm, and the old
  `.25` pre-clamp fraction. Pre-clamp fraction is the fraction of the old
  `.25` tensor outside `[0,1]` before the frozen clamp. No target or candidate
  outcome may define a group.
- Every group/tail report retains its planned denominator, missing/nonfinite
  count, and zero/tie rule. A0D is descriptive: only schema, hash, row
  completeness, and finite-value checks can pass or fail; it cannot rank or
  select an A0P method/window/state.

### A0P Population And Common Quantities

- A0P uses all 256 retained A0R `r1` pre-update states from epochs 9-16. Each
  state restores the exact repair-head, AdamW, RNG, update index, and source
  manifest hash before every factor cell.
- Every state evaluates the complete Cartesian design of three methods and
  three windows. Each post-step state is rendered against all heldout128 images
  under both `D_ref` and `D_rep`; that heldout bank cannot participate in
  proposal construction, projection, backtracking, or stopping.
- Let `g` be the rendered `.25` MSE gradient, and let
  `c_anchor`, `c_harm`, `c_margin`, and `c_cvar` be the four v3z constraint
  gradients, in that order, on the assigned construction window. Gradients and
  active-set calculations use float64. The original v3z source SHA and all
  state/source manifests must match before a cell is eligible.
- AdamW is fixed to learning rate `5e-4`, weight decay `1e-5`,
  betas `(0.9,0.999)`, epsilon `1e-8`, `amsgrad=false`, and clip norm `0.1`.
  Its restored moments and step counter are part of the state identity.

### A0P Method Factor

1. `historical_sequential_gradient`: call the immutable v3z
   `projected_grad` implementation as numerical authority, with the four
   constraints in the recorded order, clip to `0.1`, and take one cloned AdamW
   step.
2. `exact_gradient_intersection`: solve
   `argmin_d 0.5*||d-g||^2` subject to `c_k^T d >= 0` for all four constraints.
   Enumerate all 16 active sets, normalize nonzero rows, use SVD pseudoinverse
   `rcond=1e-12`, require primal and dual residuals `<= 1e-10`, and break equal
   objective values by lexicographic active-set tuple. Clip the result to `0.1`
   and take the same cloned AdamW step. An absent feasible active-set solution,
   nonfinite value, or failed residual is an invalid cell.
3. `actual_proposal_projection_with_backtracking`: form the actual AdamW
   displacement `p0=AdamW(theta,clip_0.1(g))-theta`, including restored moments,
   bias correction, and decoupled decay. Solve
   `argmin_p 0.5*||p-p0||^2` subject to `c_k^T p <= 0` with the same active-set
   solver. Test `beta` in descending order from
   `{1, 2^-1, ..., 2^-10}` using full post-clamp construction-window renders.
   Accept the first beta for which every rendered MSE/anchor/harm/margin/CVaR
   quantity `R_j` satisfies
   `R_j(post) <= R_j(pre) + 2*(1e-12 + 1e-12*max(abs(R_j(pre)),abs(R_j(post))))`.
   If none passes, record exact zero displacement and `backtracking_null`; it
   remains an analyzed cell, not a dropped state.

### A0P Window Factor

- `fixed4`: the exact recorded historical state window.
- `shuffled16`: rank all update128 names by SHA-256 of
  `shuffled16|state_sha256|name` and take the lowest 16.
- `prestratified32`: compute the initial inherited-harm ratio per image as the
  worst-operator `H_inherited/max(old_125_mse,1e-30)`, sort by `(ratio,name)`,
  form eight consecutive strata of 16, then choose four per stratum by
  SHA-256 rank `prestratified32|state_sha256|stratum|name`.
- Materialize and hash every assignment before the first shadow outcome is
  evaluated. A missing name, overlap error, or wrong cardinality fails the
  affected cell and the complete-family gate.

### A0P Endpoints, Bootstrap, And Trigger

- Store actual parameter displacement, each `c_k^T p`, KKT residual, active
  set, displacement norm, beta, and full heldout render metrics. The post-step
  safety endpoints are anchor, harm, margin, CVaR25, CVaR10, severe/hard
  counts, and rendered MSE; utility endpoints are mean and p05 PSNR.
- Use exactly 1,000 paired two-way bootstrap replicates with
  `numpy.random.Generator(numpy.random.PCG64(3407))`: resample 256 state IDs
  and heldout128 image IDs independently with replacement while retaining both
  paired operators and all nine cells. Build simultaneous 95% max-statistic
  bounds over all cell effects, both non-historical versus historical contrasts,
  and method-by-window difference-in-differences. Do not bootstrap cells
  separately and combine their intervals.
- Utility non-inferiority is fixed at `-0.005 dB`. Safety is smaller-is-better;
  utility is larger-is-better. Unrounded float64 comparisons govern ties.
- A proposal/window is positive only if the complete family is present, every
  solver is valid, simultaneous safety UCBs are within the frozen numerical
  tolerance both absolutely and versus historical for both operators, at least
  one harm or CVaR25 UCB is strictly lower than minus that tolerance, and
  simultaneous worst-operator mean-PSNR LCBs are at least `-0.005 dB` both
  absolutely and versus historical.
- Multiple positive proposal windows are ordered by lowest worst safety UCB,
  then `fixed4`, `shuffled16`, `prestratified32`. Their only typed outcome is
  `A0P_ACTUAL_PROPOSAL_POSITIVE_R3_HANDOFF`. No positive proposal produces an
  automatic A0M authorization. If no nonhistorical method is positive, emit
  `A0P_NO_LOCAL_CORRECTION_R3_HANDOFF`. Exact-gradient-only positivity,
  interaction reversal, incomplete family, or numerical failure produces
  `A0P_INCONCLUSIVE_AMENDMENT_REQUIRED`.
- Any missing, unpaired, infeasible, nonfinite, hash-mismatched, or
  unverifiable cell is `FAIL_CLOSED`; no fallback may drop cells, replace actual
  renders, or alter method/window definitions after outcomes exist.

## Decision

- Decision label: `V4A_A0R_PASS_A0DP_AMENDMENT_FROZEN_R2_IMPLEMENTATION_COMPLETE_AWAITING_R1_PREFLIGHT`.
- Image/global metric reason: historical global utility/safety disagreement needs exact per-image reconstruction before any redefinition or new model claim.
- Mechanism reason: actual AdamW displacement and projection-order effects are currently unobserved.
- Preservation or regression reason: low-haze and tail safety remain unresolved; the old `.125` anchor is retained without relaxation.
- Inherited harm versus anchor: A0D will quantify historical old `.25` burden separately from v3z repair-added burden.
- Candidate total harm versus anchor: A0D will report new `.25` burden against old `.125` on the same rows.
- Intervention-added harm versus predecessor: A0D will report the new repair contrast against old `.25` without using it to excuse total harm.
- Group/split/seed uncertainty and interaction reason: historical fixed slices and fixed4 windows are diagnostic; A0P pairs states/windows and retains the update-method interaction.
- Evidence role and independence reason: every v4a result is engineering/debug or development evidence; no candidate confirmation is claimed.
- Cost/deployability reason: v4a adds cloud diagnostic storage only and has no deployable inference artifact.
- What this decides next: the amended A0D/A0P tools are statically validated.
  Cloud preflight and launch require their own dynamic authorization;
  an A0P closeout can emit only a new R3 handoff.
- Typed closeout path: `RUN_ROOT/a0r-run-id/v4a_a0r_closeout.json` staged as compact evidence after review.
- `PASS` authorizes: `A0D_AND_A0P_ONLY`.
- `INCONCLUSIVE` authorizes: `NO_PROMOTION; WRITE_R3_REVIEWED_AMENDMENT_ONLY`.
- `FAIL` stops: `A0P_A0M_A1_V4B_V4C_CANARY_AND_LOCKED_TEST`.

## A0P Result And R3 Decision

A0P `a0p_r4` completed the full frozen family: 256 retained states, 2,304
method/window cells, 128 heldout images, both operators, 655,360 raw rows, and
the joint 1,000-replicate max-statistic bootstrap. Every solver was valid,
there were no missing/nonfinite cells, and no backtracking cell collapsed to a
null step. The typed closeout is `COMPLETED_GATE_PASS` with decision
`A0P_NO_LOCAL_CORRECTION_R3_HANDOFF`.

The R3 interpretation is mechanism-specific:

- exact common-intersection projection was numerically indistinguishable from
  the historical sequential projection. Across windows and operators, its
  harm effect versus historical was approximately `-2.1e-12` to `-5.7e-12`
  and its CVaR25 effect approximately `-7.7e-12` to `-2.0e-11`; every
  simultaneous interval crossed zero;
- actual-proposal projection changed the applied displacement and preserved
  the `-0.005 dB` utility non-inferiority line, but its harm, CVaR25, and
  CVaR10 point effects versus historical were all positive for every window
  and both operators: about `+2.08e-10` to `+2.26e-10`, `+7.32e-10` to
  `+7.98e-10`, and `+1.08e-9` to `+1.17e-9`, respectively;
- fixed4, shuffled16, and prestratified32 did not produce a positive method or
  a sign-reversing interaction. Window estimation and projection order are
  therefore not supported as the material v3z heldout-safety bottleneck;
- no method/window had a simultaneous harm or CVaR25 UCB strictly below the
  historical method. Discrete severe/hard intervals also crossed zero, but
  this did not hide a positive continuous harm/CVaR result.

This closes A0M and any optimizer/window retuning continuation. The only
authorized continuation is a new, train-derived, privileged A1F route that
tests v3z-aligned bounded `Delta-u` action-space feasibility under the same
old `.125` anchor, old `.25` predecessor, heldout128 population, and paired
`D_ref`/`D_rep` render contract. It must first prove metric/source alignment
with v3r/v3z and must not repeat generic alpha/block16 oracle questions already
answered by v3j/v3l/v3m. A1F may authorize only a separately frozen
representation-sufficiency audit or terminal learned-repair closeout.

Decision:
`V4A_A0P_NO_LOCAL_CORRECTION_AUTHORIZE_A1F_METRIC_ALIGNED_FEASIBILITY_ONLY`.

Canary, locked test, candidate selection, A0M, optimizer retuning, risk-window
search, policy replay, and direct v4b/v4c training remain forbidden.
