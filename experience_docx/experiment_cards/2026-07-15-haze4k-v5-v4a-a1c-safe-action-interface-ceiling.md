# Haze4K v5 CHD-RM v4a-A1C Safe-Action Interface Ceiling

Date: 2026-07-15

Status: `PLANNED`

## Scope

- Project: ConvIR-B Haze4K CHD-RM v5.
- Model family: privileged, no-learning replay of the frozen v4a safe bounded Delta-u action target through full-resolution and two half-resolution interfaces; no deployable candidate.
- Dataset or task: the exact A1R fresh512 train-derived names at indices 256:768, already assigned `development_screening`; no A1F rows, unused names, confirmation, canary, or locked-test access.
- Primary objective: determine whether the exact half-resolution correction interface used by A1R, or the same interface with anti-aliased downsampling, retains enough of the full privileged safe-action ceiling to support a later representation-to-action design.
- Main metric: paired-image bootstrap lower bound of each half-interface worst-operator mean safe gain over the privileged shrink oracle, jointly gated with draw-wise retention of the full privileged reference.
- Secondary metrics: full-reference gain and repairable fraction, interface repairable fraction, anti-aliased-minus-exact gain, interface endpoint error, full-to-half spectral loss diagnostics, safety excess, severe/hard regressions, action-grid choice, runtime, and peak memory.
- Execution environment: local WSL for editing and syntax/static checks only; any future runtime is restricted to `convir-4090`.
- GitHub rules commit: `d502a990fb8766d6b0b8e57ed913d1a2200297de`.
- Authoritative experiment evidence: `experience_docx/experiment_logs/haze4k_v5_chd_rm_v4a_a1f_deltau_action_feasibility_20260714/v4a_a1f_r3_review.json`, `experience_docx/experiment_logs/haze4k_v5_chd_rm_v4a_a1r_representation_sufficiency_20260714/v4a_a1r_formal_bootstrap_summary.json`, and `experience_docx/experiment_logs/haze4k_v5_chd_rm_v4a_a1r_representation_sufficiency_20260714/v4a_a1r_r3_review.json` from that GitHub rules commit.
- Local WSL path, if used for editing/static checks: `/home/ubuntu/workspace/ConvIR-B-v4a-a1c-safe-action-interface-ceiling-20260715`.
- GitHub route branch and source commit: `codex/haze4k-v5-v4a-a1c-safe-action-interface-ceiling-20260715` from `github/main@d502a990fb8766d6b0b8e57ed913d1a2200297de`; scientific replay sources remain pinned separately below.
- Cloud `REMOTE_REPO`: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v4a-a1c-haze4k_v5_chd_rm_v4a_a1c-a1c20260715-400bfecd291cd158` as deterministically derived by schema-v2 from repo name `ConvIR-B-v4a-a1c` and workspace id `a1c20260715`.
- Cloud `RUN_ROOT`: `/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v4a_a1c_safe_action_interface_ceiling_20260715`.
- Cloud `EVID_STAGE`: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v4a-a1c-haze4k_v5_chd_rm_v4a_a1c-a1c20260715-400bfecd291cd158/experience_docx/experiment_logs/haze4k_v5_chd_rm_v4a_a1c_safe_action_interface_ceiling_20260715`.
- Explicit cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.

## Agent Execution Routing

Use `MODEL_AGENT_COST_ROUTING_PROTOCOL.md` for the canonical role table and dispatcher mechanics.

- Host identity mode: `dispatcher_receipt`.
- Task-scoped host pin: `frontier/xhigh` from dispatcher handoff SHA-256 `33bd63cebce8207927da131ccfdd67bca89022ef88c14954bfe15fa74b3fbfd5` for this R3 contract-design task only.
- R3 target effort and rationale: `xhigh`: the task jointly freezes the interface estimand, reused-data claim limit, paired oracle construction, multiplicity rule, branch gates, and later-stage authority after reconciling A1F and A1R.
- Whole-task batching plan: one current R3 contract package; one later R2 runner/static package; one exact-tuple R1 S0 package; one R2 compact S0 closeout commit; one exact-tuple R1 formal start/short-observation package; one persistent R0 observation package only if the formal stage remains healthy beyond that window; one R3 interpretation package; and one R2 unchanged-verdict archive package. No package may cross an engineering repair, scientific decision, or authorization boundary.

| Applicable scope | Task class | Minimum role | Routing basis/ref | Boundary action |
| --- | --- | --- | --- | --- |
| scientific design / gate contract | `R3_SCIENTIFIC_AUTHORITY` | `frontier` | `dispatcher_classification`; received handoff above | current dispatched task; stop after validated setup commit |
| workspace / runner engineering | `R2_ENGINEERING_CONTROL` | `balanced` | `dispatcher_classification`; this card and initial authorization at the setup commit | one `task_routing` package for runner implementation and syntax/static checks only |
| authorized S0 or formal start / short observation / compact fetch | `R1_BOUNDED_EXECUTION` | `fast` | `typed_handoff`; exact committed initial authorization or S0 closeout tuple | one bounded package per committed authorization; no repair or interpretation |
| healthy long-run observation | `R0_READ_ONLY` | `fast` | `typed_handoff`; receipt from the exact formal start | one persistent task for all healthy windows; stop before repair or interpretation |
| result interpretation / terminal verdict | `R3_SCIENTIFIC_AUTHORITY` | `frontier` | `typed_handoff`; formal closeout at the then-current route commit | new `task_routing` package before any scientific branch token is issued |
| unchanged-verdict archival / sync | `R2_ENGINEERING_CONTROL` | `balanced` | `typed_handoff`; committed R3 review | one `major_handoff` package after the verdict; no verdict changes |

The current task does not use `dispatch=not_amortized`. Each later package is an
independent durable boundary whose cheaper qualified role avoids carrying the
full R3 design context. Actual dispatcher requests must replace the descriptive
references above with the exact GitHub commit and path then in force.

## Baseline Contract

- Baseline implementation: full privileged target and safe-grid helpers from v4a-A1F route commit `b30ec5d6e1c911e5687bc64f8c9f8b5e0357c660`, with the exact A1R half-resolution correction transport from route commit `49816258fbb8c56298b14ab53793d46945c975f9`.
- Baseline checkpoint or initialization: A0R epoch16/update512 final state SHA-256 `ed0832f220996af3fd8e617b7d04d643dc6ca052a3603adee99d59e78fd1e125`; no new state is initialized or learned.
- Evaluation entrypoint: reserved for the later R2 package as `experience_docx/tools/chd_rm_v4a_a1c_safe_action_interface_ceiling_v2.py`, invoked only through `experience_docx/tools/run_chd_rm_v4a_a1c_safe_action_interface_ceiling_v2.sh`. The inherited draft evaluator/runner are not referenced or authorized by this revised contract.
- Training entrypoint: not applicable: training, optimization, fitting, calibration, and weight updates are forbidden.
- Dataset and split: exact A1R fresh512 names/folds from indices 256:768 of the frozen 1,200-name list; 57 names use the 208x208 half lattice and 455 use 240x320, with all 512 scored once without exclusion.
- Preprocessing and decoding: immutable v3z Haze4K load/pad, frozen base/control/gate, D_ref/D_rep, old support and bounds, float32 add-and-clamp renderer, and A1F 65-point shrink/direction grids.
- Metric implementation: rebuild full/current target endpoints once per image/operator; apply the three frozen interface cells; use identical privileged safe selection, canonical RGB MSE/PSNR, and paired image resampling for every cell.
- Reproduced baseline result: A1F established safe bounded direction headroom; A1R fresh512 then reported primary gain LCB95 `+0.0152048253 dB` and oracle-retention LCB95 `0.0923048771`, below `+0.020 dB` and `0.25`, while its full privileged denominator remained materially positive.
- Known reproduction gap: no source, name, shape, target, grid, renderer, or full-reference drift is accepted. A1C S0 or formal stops before a scientific result if the pinned identities or exact A1R bilinear transport cannot be reconstructed.
- Reference entrypoints that must remain stable: v3z `candidate_predictions`, v3w frozen sample/operator path, A1F `grid_metrics` and safe selection, A1R fresh split/source manifest, A1R context-size derivation, A0R final state, and v3j support/bounds.
- Checkpoint/export/resume contract: no checkpoint, export, learned state, or resume; interruption requires a new run id and output directory.

## Most Valuable Attempt

- Why this is the highest-value next attempt: A1F proves safe direction headroom while A1R recovers only about one tenth of its full oracle denominator; an ideal replay through the same half-resolution output interface isolates whether spatial transport itself caps the learner.
- Target failure or opportunity: separate half-resolution interface loss from representation/learner loss without training another probe or changing the action family.
- Cheap preflight evidence: exact source/name/shape inventory plus deterministic full, exact-half, and antialiased-half endpoint construction on the first 32 A1R names.
- Earliest decisive gate: the full privileged reference must first reproduce material safe headroom; then at least one preregistered half interface must clear the multiplicity-controlled `+0.080 dB` absolute-gain, `0.50` worst-size retention, and `0.20` repairability floors.
- Expected cost or attempt-count saving: one frozen forward per image/operator and tensor-only grid replay can reject an inadequate interface before any representation, model, selector, or policy implementation.
- What success decides: which, if any, half-resolution interface is scientifically adequate for a separately preregistered representation-to-action route design.
- What failure decides: an interface with its adjusted upper gain bound below `+0.020 dB` is closed; if full passes but both half interfaces fail, later work must change action resolution/transport rather than repeat an A1R-like learner at half resolution; if full fails, no aligned A1C continuation exists on fresh512.
- Why a cheaper diagnostic is not enough: endpoint interpolation error or spectra alone do not measure rendered safe utility, and A1R probe performance confounds representation, fitting, and interface transport.

## Hypothesis

- Observed failure: A1R contains real target-direction signal but fails material gain and full-oracle retention under a half-resolution learned correction interface.
- Target mechanism: bilinear downsample/replay may remove action-critical spatial direction content before any learner can express it; anti-aliasing may preserve a safer low-frequency correction than the exact historical downsample.
- Null hypothesis: neither fixed half-resolution interface retains the preregistered absolute safe gain, full-reference retention, and repairable coverage on fresh512.
- Preferred causal hypothesis: an ideal correction replayed through the exact A1R half interface remains materially useful, so representation-to-action prediction rather than interface resolution is the active A1R bottleneck.
- Competing hypothesis or confound: the exact half interface is inadequate but the anti-aliased variant is adequate, or both are inadequate despite a valid full reference, making interface transport/resolution a material bottleneck.
- Cheapest observation that separates them: paired privileged safe-grid replay of one full endpoint and two deterministic half-resolution endpoint transports on the same image/operator units.

Mechanism sentence:

```text
For the reused A1R fresh512 image/operator units, an ideal target correction
transported through the exact or antialiased half-resolution interface should
retain material safe gain relative to the full privileged target if the A1R
failure lies in representation-to-action prediction rather than interface
resolution; a full-reference pass with both half interfaces below gate favors
an interface bottleneck instead.
```

## Estimand And Risk Attribution

- Target population: the fixed A1R fresh512 development population only; reuse after A1R prevents any confirmation or broader Haze4K claim.
- Analysis unit and grouping unit: clean-reference image, with D_ref/D_rep and all three interface cells paired within image.
- Intervention or factor contrast: `full_privileged`, `exact_half_replay`, and `antialiased_half_replay` endpoint interfaces, each evaluated as a safe direction-union oracle against the same privileged shrink oracle.
- Reference/direct predecessor: privileged shrink is the absolute-utility reference; `full_privileged` is the retention denominator; old `.25` is the predecessor safety reference and old `.125` is the common anchor.
- Outcome, direction, and aggregation: higher rendered PSNR is better; the operator-specific image mean is computed first and the worse paired operator decides each bootstrap draw.
- Claim type: causal
- Identification assumptions and sensitivity limits: all cells share source tensors, targets, grids, safety constraints, labels, ordering, and renderer and differ only in the frozen endpoint transport. Privileged labels select actions, so the claim is a nondeployable interface ceiling on reused development evidence.
- Minimum worthwhile effect or risk limit: full-reference global worst-operator LCB95 gain over shrink `>=+0.080 dB`, every native-size/operator full-reference gain LCB95 `>0`, and repairable LCB95 `>=0.20`; each adequate half interface requires multiplicity-controlled worst-operator LCB of gain `>=+0.080 dB`, worst-native-size draw-wise full-reference retention `>=0.50`, and repairable fraction `>=0.20`.
- Equivalence/non-inferiority margin and independent source, if claimed: no equivalence claim is made. The pre-execution A1C draft already archived on `github/main@d502a990...` fixed `+0.080 dB` as the smallest actionable interface value, `0.50` as material full-reference retention, and UCB95 `<+0.020 dB` as interface closure; this review corrects the estimand orientation without weakening those margins. The `0.20` repair floor and descriptive anti-aliased-minus-exact LCB95 of `+0.005 dB` come from the pre-A1R contracts and cannot authorize a branch alone.
- Common safety anchor: old `.125` on the same image/operator.
- Inherited-harm estimand: `max(MSE(old_.25)-MSE(old_.125),0)`.
- Candidate-total-harm estimand: for each interface oracle, `max(MSE(selected_.25)-MSE(old_.125),0)`.
- Intervention-added-harm estimand: candidate total harm minus inherited harm, with every selected cell additionally constrained to be non-worse than old `.25` within the frozen numerical tolerance.

## Design And Identifiability

- Design type: `paired_ablation`
- Why this is the cheapest design that identifies the estimand: all three deterministic endpoint cells reuse one frozen forward and differ only in a tensor transport; no learned probe, fold fit, candidate, or new data source is required.
- Experimental unit and randomization/pairing: fixed image groups; every image/operator supplies all cells, shrink reference, full denominator, grid values, and bootstrap contributions. No randomized assignment is needed for the deterministic transport contrast.
- Blocking, exclusion, failure, and missing-cell policy: exact full/half spatial shape and historical outer fold are nuisance blocks; no row may be excluded. Missing names/operators/cells, nonfinite values, shape mismatch, source drift, or safety violation fails structural integrity.
- Formal subgroup definitions and pre-intervention/independent source: the two native shape blocks, historical A1R fold, and paired operator are frozen by the A1R source manifest; subgroup results are sensitivity diagnostics and cannot alter the gate.
- Primary comparison family and multiplicity treatment: two half-interface adequacy candidates can authorize later design. Bonferroni allocates one-sided alpha `0.025` to each candidate; within a candidate, gain, worst-native-size retention, and repairability form an intersection-union gate, so all three one-sided 97.5% lower bounds must pass and no additional within-candidate adjustment is needed. This controls familywise false adequacy at `<=0.05`.
- Fractional-design resolution and alias structure, if applicable: not applicable: all three fixed interface cells are complete and paired.
- Negligible-interaction assumptions and targeted de-alias follow-up: no shape-by-interface or operator-by-interface interaction is assumed negligible; both operators decide through the worst rule and both shape blocks are reported.
- Paired seeds/folds/data order/evaluation operators: PCG64 seed `3407`, exact A1R name/fold/order, identical grid/tie rules, and paired D_ref/D_rep for every cell.
- Natural groups and repeated grouped-split or leave-one-group-out plan: image is the natural group; no split is needed because nothing is fit. Historical four-fold labels are retained only for fold sensitivity and provenance.
- Split/seed uncertainty required for the claim: all 512 images and fold summaries are required; one deterministic bootstrap seed limits the result to this fixed development screen.
- Uncertainty estimator and dependence/group structure: 4,000 paired PCG64(3407) image bootstrap draws resample complete cell/operator rows; the worst operator is computed within each draw. Retention is computed separately inside each native-size stratum as candidate worst-operator gain divided by full-reference worst-operator gain in the same draw, and the worse stratum decides. Any nonpositive full denominator makes retention non-estimable and the result at most inconclusive.
- Sample/group/split/seed count justified by power or target interval width: all available A1R fresh512 names are mandatory and provide the same population that generated the A1R failure; no post-result sample expansion is allowed in this route.
- Fixed-data attainable precision or smallest reliably detectable effect: the 97.5% candidate bounds are the attainable family-controlled precision. A candidate passes only at `+0.080 dB` / `0.50` / `0.20`; it fails only when an adjusted upper gain bound is `<+0.020 dB`, an adjusted upper retention bound is `<0.50`, or an adjusted upper repairability bound is `<0.20`; every other non-pass is `INCONCLUSIVE`, never equivalence.

Frozen interface hierarchy:

| Cell | Low-resolution operator | Reconstructed endpoint | Scientific role |
| --- | --- | --- | --- |
| `full_privileged` | none | `target_delta` on the full action lattice | full privileged reference and retention denominator |
| `exact_half_replay` | `D0(x)=interpolate(x,size=context_shape,mode=bilinear,align_corners=False,antialias=False)` | `clip(current_delta + support*U(D0(target_delta)-D0(current_delta)), +/-B)` | exact idealized A1R correction interface |
| `antialiased_half_replay` | `Daa` equals `D0` except `antialias=True` | `clip(current_delta + support*U(Daa(target_delta)-Daa(current_delta)), +/-B)` | isolated anti-aliasing alternative |

For both half cells, `U` is bilinear interpolation to the full current-Delta-u
shape with `align_corners=False` and no antialias option. The exact A1R replay
pairs are 400x400 -> 208x208 -> 400x400 for 57 names and
480x640 -> 240x320 -> 480x640 for 455 names; the first low lattice comes from
the padded context and must not be replaced by an assumed 200x200 half size.
`target_delta`, `current_delta`, support, bounds, float32 dtype, the
65-point current-to-endpoint grid, shrink union, safe selector, and tie order
are otherwise identical. The low correction must lie within the theoretical
`+/-2B` endpoint-difference range; a violation is structural failure, not a
silent clip. Interface endpoints are clipped only where A1R clips them.

## Change

- Code branch: `codex/haze4k-v5-v4a-a1c-safe-action-interface-ceiling-20260715`.
- Exact code/config change: this task adds only the frozen route card, schema-v2 operations projection, and typed initial authorization. A later R2 package may implement the reserved no-training audit and runner verbatim from this contract.
- Enabled mechanisms: full privileged target replay, exact A1R half correction transport, anti-aliased half correction transport, fixed safe-grid selection, paired grouped bootstrap, and explicit interface adequacy classification.
- Explicitly disabled mechanisms: probe/model/operator/current-head updates, training, optimization, fitting, calibration, target/grid/bound/support changes, candidate or policy selection, confirmation, canary, and locked test.
- Parameter/runtime/memory impact expected: zero deployable parameters and no retained state; one frozen model pass per image/operator plus three tensor-only direction grids.
- Initialization or no-op behavior: grid zero must be bitwise identical to the canonical shrink/predecessor tensor for all cells; a zero correction returns current Delta-u exactly.
- Resume policy: none; an interrupted operation uses a fresh output id after an R2 integrity review.
- Defaults changed: only the diagnostic endpoint transport factor is added; the antialiased downsample is an explicit cell and never becomes a default.
- Defaults intentionally preserved: frozen sources/assets/state, fresh512 names, shape blocks, operators, support, bounds, renderer, action grids, anchor, safety selector, tie rule, seed, and metric code.

## Preflight

| Check | Pass line | Result |
| --- | --- | --- |
| route/rules identity | exact route branch from `d502a990...`; clean dedicated workspace; rules commit equals current `github/main` | frozen in this setup; recheck at R2 and before S0 |
| parent scientific authority | A1R R3 review is `COMPLETED_R3_REVIEW`, decision closes the current probe family, and authorizes only a separate R3 amendment/design | verified from `github/main`; A1C remains a separate no-learning design |
| source/state identity | A1F helper commit, A1R helper commit, A0R state SHA, fresh split SHA, and asset hashes match the named GitHub evidence | contract frozen; runner-integrated recheck required |
| interface identity | exact 400x400/208x208 and 480x640/240x320 full/context shape pairs; `D0`, `Daa`, `U`, dtype, support order, endpoint clamp, and theoretical correction bounds match the formulas above | contract frozen; R2 reference checks and S0 required |
| authorization boundary | initial JSON exact tuple permits only S0; formal requires the committed S0 pass tuple | frozen in `route_operations.json` |
| runner availability | the reserved `v2` runner and entrypoint do not exist in this setup commit; inherited draft runtime files are explicitly outside this revised manifest | expected blocker until the bounded R2 implementation package completes |
| GPU capacity | a future selected GPU has at least 18,000 MiB free and at most 10% utilization | dynamic cloud preflight only; no cloud contact in this task |

## Mechanism Metrics

| Metric | Why it matches the route | Gate subset | Final artifact |
| --- | --- | --- | --- |
| full safe gain over shrink and repairable fraction | establishes an estimable fresh512 privileged denominator | full512, worst operator | compact bootstrap summary |
| half-interface safe gain over shrink | measures absolute usable interface ceiling | full512, each candidate, worst operator | compact bootstrap summary |
| draw-wise half/full retention | isolates loss relative to the paired full target | full512, each candidate, worst operator | compact bootstrap summary |
| anti-aliased-minus-exact safe gain | tests whether filtering changes material rendered value | paired full512; diagnostic LCB95 | compact bootstrap summary |
| endpoint MSE/cosine and high-frequency energy loss | connects transport distortion to rendered utility without becoming a gate | cells, operators, shape blocks | compact interface summary |
| total/added harm, safety excess, severe/hard counts | preserves the common-anchor risk contract | every selected row | compact cell/operator summary and closeout |

## Controls

| Control | Purpose | Pass line |
| --- | --- | --- |
| exact A1R fresh source and current state | exclude split/state drift | pinned hashes, 512 unique names, both operators, and exact shape counts |
| full A1F target/grid replay | establish the parent target and denominator | full target/support/bounds and grid/tie identities exact; all rows finite |
| zero/no-op path | exclude interpolation or batched-grid drift | zero rendered tensors bitwise equal to canonical predecessor for every cell |
| exact historical half replay `D0/U` | isolate ideal prediction from transport | scalar and batched compositions agree within frozen float32 tolerance; no implicit antialias |
| antialiased half replay `Daa/U` | change only the downsample filter | all other tensors, shapes, grid values, safety rules, and tie ranks identical |
| paired operators and shape blocks | prevent an easy operator/size from masking failure | complete D_ref/D_rep cells; worst operator decides; both shape blocks reported |

## Evidence-Role Ledger

| Evidence source or groups | Role | Allowed uses | Forbidden uses |
| --- | --- | --- | --- |
| first 32 A1R fresh names | `engineering_debug` | source, shape, no-op, interpolation identity, finite, bound, and safety checks only | metric calibration, candidate choice, or scientific gate |
| exact A1R fresh512 indices 256:768 | `development_screening` | fixed A1C interface ceiling and mechanism contrasts | confirmation, promotion, threshold fitting, candidate/policy training, or sealed claim |
| A1F/A1R compact GitHub evidence | historical parent evidence | source identity, threshold provenance, target/interface contract, and failure context | row-level fitting, post-result threshold changes, or rescue selection |
| A1X train-derived indices 768:1200 | `confirmation` | none in A1C; retained untouched for a separately designed nested accessibility/deployability question | any A1C read, cache, expansion, threshold, or rescue use |
| v3j controller calibration and route-confirm groups | `confirmation` | none in A1C | all A1C access or reuse |
| Haze4K confirmation/canary and locked test | `sealed_final` | none in A1C | all access, inspection, tuning, selection, or evaluation |

- Candidate/threshold/operator freeze point: all three interfaces, formulas, sizes, cells, grids, safety/tie rules, thresholds, candidate multiplicity allocation, operators, and bootstrap semantics freeze in this card before any runner is implemented.
- Independent confirmation contract: none; A1C reuses A1R development evidence and can authorize only a new R3 route design, never a candidate, training run, policy replay, or confirmation claim.
- Nested group-respecting resampling contract, if no separate confirmation set: not used for confirmation; the paired image bootstrap quantifies only this fixed development screen.
- Final sealed-use authorization and one-use policy: no sealed or confirmation use is authorized.
- Post-sealed rule (`report/close only; no tuning or reselection`): not applicable because all sealed evidence remains untouched; any future sealed route must freeze its complete contract separately.

## Fair Run Contract

- Training or inference budget: S0 uses the first 32 fresh names and all cells/operators for integrity only; formal uses exactly 512 names, two operators, one frozen forward per image/operator, one shrink grid, and three direction-interface grids.
- Batch/sample policy: one image/operator source forward at a time; candidate grids may be chunked only if scalar and batched float32 results meet the frozen tolerance. No resize beyond the named interface operations, crop, padding change, exclusion, substitution, or augmentation.
- Optimizer: not applicable: no trainable state or update exists.
- Schedule: S0 only under the initial authorization; formal only after a committed S0 closeout with the exact pass tuple; then stop for R3 review.
- Loss weights: not applicable: labels are used only by the fixed target and privileged safe selector.
- Random seed policy: deterministic source/grid order; PCG64 seed `3407` for 4,000 paired bootstrap draws only.
- Evaluation cadence: one terminal integrity closeout for S0 and one terminal scientific closeout for formal; no interim scientific inspection.
- Checkpoint cadence: not applicable: no new state exists.
- Hardware/runtime assumptions: future runtime only on `convir-4090` with the explicit cu121 Python and one dynamically qualified CUDA GPU; CPU results cannot replace it.
- Allowed resume behavior: none; fresh output id and explicit engineering classification are required after interruption.
- Sample-size policy: S0 exactly 32 unique names; formal exactly 512 unique names, 1,024 image/operator units, all three interface cells plus shrink, and no missing row.
- Dependency/version assumptions: exact A1F/A1R route commits and source hashes; PyTorch from the explicit environment must support explicit `antialias=False/True` bilinear downsampling with `align_corners=False`.
- Selected decision profile: `feasibility_oracle`
- Learned-state retention required: no: the route creates no learned, optimized, selected, or resumable state.
- Omitted or specialized stage rationale: no probe training, policy replay, confirmation, canary, or locked test can identify the pure endpoint-interface ceiling; only integrity smoke and one formal development replay are included.
- Integrated pre-smoke contract: the future tracked runner must verify route/runner/source hashes, authorization tuple, 512-name identity, exact shape counts, cell formulas, no-op, finite tensors, correction/support/bound limits, scalar-versus-batched grid equivalence, forbidden-data flags, output collision, and heartbeat setup before S0 work.
- Expected wall-time budget and required phase timings: R2 must refine estimates only from static implementation analysis before S0; conservative hard budgets are S0 `<=20` minutes and formal `<=90` minutes, with setup, source-forward, interface-grid, bootstrap, summary, and closeout timings reported separately.
- Heartbeat cadence, stale threshold, and monitor profile: future runner heartbeat at least every 60 seconds; S0 uses `short`, formal uses `standard`, and both use a 300-second stale threshold.
- Maximum model-visible observations and escalation condition: one start receipt and at most nine S0 or twenty-one formal status windows in their respective persistent tasks; stale heartbeat, missing closeout, source mismatch, collision, nonfinite result, or budget overrun stops for engineering escalation without scientific interpretation.
- Workspace policy (`fresh_route` or `exact_continuation`) and rationale: S0 uses `fresh_route` for the first schema-v2 cloud workspace; formal uses `exact_continuation` only after a clean same-branch fast-forward commit containing the validated S0 closeout. Neither policy permits reuse of an older route or dirty workspace.

## Gates

| Stage | Estimand/question | Evidence role and budget/sample scope | Gate type, threshold source, and multiplicity rule | `PASS` authorizes |
| --- | --- | --- | --- | --- |
| S0 interface integrity | Are the pinned source, fresh32 identities, full/half shapes, three endpoint transports, grids, no-op, bounds, safety, and forbidden-data flags exact and finite? | `engineering_debug`, first 32 fresh names, both operators and all cells | structural binary gate: exact identities/counts/formulas; zero/no-op exact; no nonfinite, theoretical correction-bound, support, endpoint-bound, anchor, predecessor, severe, or hard violation; no scientific metric inspected | `A1C_FORMAL_INTERFACE_CEILING_ONLY` through the exact committed closeout tuple |
| A1C full reference prerequisite | Is the fresh512 full privileged target a material and broad safe denominator? | `development_screening`, exact fresh512, both operators and native-size strata | scientific utility plus safety: 4,000 paired draws; global worst-operator LCB95 gain over shrink `>=+0.080 dB`, every native-size/operator gain LCB95 `>0`, and repairable LCB95 `>=0.20`; all rows anchor/predecessor non-worse; zero severe/hard regression. UCB gain `<+0.020 dB`, nonpositive native-size/operator UCB, or repairability UCB `<0.20` is `FAIL`; a remaining non-pass is `INCONCLUSIVE` | no direct stage; permits evaluation of the half-interface adequacy family in the same frozen formal result |
| A1C half-interface adequacy family | Does exact half or antialiased half retain material full-reference safe value? | `development_screening`, exact fresh512, paired cells/operators/size strata | two candidate IUTs with Bonferroni alpha `0.025` each: all one-sided 97.5% LCBs for gain `>=+0.080 dB`, worst-size paired full retention `>=0.50`, and repairability `>=0.20`; candidate `FAIL` requires adjusted UCB gain `<+0.020 dB`, retention `<0.50`, or repairability `<0.20`; otherwise it is `INCONCLUSIVE`. Safety is a hard guard. Anti-aliased-minus-exact LCB95 `>=+0.005 dB` is mechanism-only | formal closeout authorizes `R3_REVIEW_ONLY`; no implementation, training, policy, candidate, canary, or locked test |
| independent confirmation | not part of A1C | none | prohibited because fresh512 is reused development evidence | none |
| sealed final | not part of A1C | none | prohibited | none |

Formal typed outcome mapping before R3 interpretation:

| Frozen outcome | Formal decision token | Formal authorizes |
| --- | --- | --- |
| full reference passes and exact half passes | `V4A_A1C_EXACT_HALF_INTERFACE_ADEQUACY_PASS_R3_HANDOFF` | `R3_REVIEW_ONLY` |
| full passes, exact does not pass, and antialiased half passes | `V4A_A1C_ANTIALIASED_HALF_INTERFACE_ADEQUACY_ONLY_PASS_R3_HANDOFF` | `R3_REVIEW_ONLY` |
| full passes and both half candidates fail | `V4A_A1C_HALF_INTERFACE_ADEQUACY_FAIL_R3_HANDOFF` | `R3_REVIEW_ONLY` |
| full reference or a candidate family is statistically unresolved | `V4A_A1C_INTERFACE_RESULT_INCONCLUSIVE_R3_HANDOFF` | `R3_REVIEW_ONLY` |
| full reference fails or structural/safety validity fails | `V4A_A1C_FULL_REFERENCE_OR_INTEGRITY_FAIL_R3_HANDOFF` | `R3_REVIEW_ONLY` |

## Analysis Plan

- Per-sample or subgroup analysis: cloud-only rows by image/operator/cell; compact cell/operator and shape/fold summaries report mean, p05/p10, repairable fraction, grid choice, endpoint error/cosine, spectral loss, and safety attribution.
- Visual or qualitative analysis: none; no images are required or permitted in compact evidence.
- Complexity analysis: report wall time, phase timings, and peak allocated GPU memory only; no deployability or latency claim.
- Robustness or held-out analysis: both operators are paired and the worse decides; both native shape blocks and four historical folds are sensitivity summaries. No subgroup rescues a failed global gate.
- Regression analysis: paired anti-aliased-minus-exact and half-minus-full effects are fixed contrasts; no post-result regression, calibration, threshold, or action search is allowed.
- Main-effect/interaction and alias analysis, if applicable: not factorial; report interface-by-operator and interface-by-shape descriptive contrasts without assuming either interaction is negligible.
- Group/split/seed uncertainty and sensitivity analysis: paired image bootstrap retains cells and operators within draw; report 95% full-reference and 97.5% candidate bounds, fold/shape point summaries, and the exact order-statistic rule.
- Screening-selection versus confirmation analysis: fresh512 reuse makes every result exploratory development screening; the better interface cannot become a candidate or confirmation result and only the frozen R3 mapping below can authorize a new design.
- Required docs to update: route card, typed S0/formal closeouts, evidence README, compact source/interface/bootstrap summaries, CHD-RM index, and global index only at a terminal R3 handoff.
- Required artifacts to retain: tracked runner/entrypoint after R2 implementation, source manifest, compact cell/operator/interface/bootstrap summaries, typed closeouts, status, hashes, and terminal R3 review.
- Required artifacts to delete or keep external: raw per-image/action rows, tensors, arrays, caches, images, full logs, and any reconstructed model state remain under cloud `RUN_ROOT` and never enter Git.
- Evidence package contents: route card, route-operations manifest, initial authorization, later runner, evidence README, compact source/interface/bootstrap summaries, typed closeouts, and R3 review.
- Evidence package audit: compact text/JSON/CSV summaries only, with no raw rows, selected-action table, feature table, checkpoint, image, array, or archive.

## Decision

- Decision label: `V4A_A1C_S0_AUTHORIZED_INITIAL_ONLY`.
- Image/global metric reason: rendered safe utility is an image-level claim; pixel errors or operator means without paired image uncertainty cannot decide the interface ceiling.
- Mechanism reason: an ideal target correction passed through the exact A1R transport removes learner error, while the antialiased cell changes only the downsample filter.
- Preservation or regression reason: every selected action must remain non-worse than old `.125` at `.125` and old `.25` at `.25`, with zero severe/hard regression, before aggregate utility is considered.
- Inherited harm versus anchor: old `.25` burden relative to old `.125` is retained for every image/operator.
- Candidate total harm versus anchor: each interface-selected `.25` burden uses the same old `.125` anchor.
- Intervention-added harm versus predecessor: each interface-selected action must add no harm beyond old `.25` within the frozen numerical tolerance.
- Group/split/seed uncertainty and interaction reason: complete paired image resampling, worst-operator-within-draw, family-controlled candidate bounds, and shape/fold sensitivity preserve the relevant dependence; no new-population or seed claim is made.
- Evidence role and independence reason: A1C deliberately reuses A1R fresh512 to diagnose that exact failure and therefore cannot supply confirmation, selection, or promotion evidence.
- Cost/deployability reason: the audit has zero deployable parameters and privileged target/action selection; it estimates only an interface ceiling.
- What this decides next: after formal closeout, a new R3 review may issue exactly one of these bounded tokens: exact-half pass -> `R3_A1X_EXACT_HALF_ACCESSIBILITY_ROUTE_DESIGN_ONLY`; exact non-pass plus antialiased pass -> `R3_A1X_ANTIALIASED_HALF_ACCESSIBILITY_ROUTE_DESIGN_ONLY`; full pass plus both half fail -> `R3_A1X_FULL_INTERFACE_ACCESSIBILITY_ROUTE_DESIGN_ONLY`; inconclusive -> `R3_A1C_AMENDMENT_DESIGN_ONLY`; full/integrity fail -> `NONE`. No token authorizes implementation, A1X data access, or runtime by itself.
- Typed closeout path: S0 `EVID_STAGE/v4a_a1c_s0_closeout.json`; formal `EVID_STAGE/v4a_a1c_formal_closeout.json`.
- `PASS` authorizes: `R3_REVIEW_ONLY`; only the reviewed mapping in the preceding field may create a later route-design token.
- `INCONCLUSIVE` authorizes: `R3_REVIEW_ONLY`; no sample expansion, threshold change, confirmation use, or runtime continuation is automatic.
- `FAIL` stops: `SAME_ROUTE_REPLAY_DIRECT_TRAINING_POLICY_CANDIDATE_CONFIRMATION_CANARY_AND_LOCKED_TEST`.
