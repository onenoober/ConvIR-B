# CHD-RM Haze4K Experiment Index

Date: 2026-07-12

Status: v3o A0 smoke passed fixed-candidate replay integrity on 32 OOF images
per frozen operator. The only current authorization is its exact 1,200-image
formal candidate-SSE audit; no policy replay, route-confirm audit, canary
expansion, locked-test access, controller training, learned ranker, or policy
deployment is authorized. v3m showed that block16 common-action oracle value
and deployable local label observability are real, but the frozen A2 calibrated
policy converts them into unsafe image-level tails. v3n then tested stricter
`.125 -> .25` false-intervention semantics and selected zero held-out blocks.

## Research Direction

```text
Continuous haze-density-aware region-adaptive residual modulation with low-haze protection for ConvIR-B Haze4K dehazing.
```

## Current Scope

- Dataset: Haze4K only.
- Backbone: ConvIR-B.
- Task: single-image dehazing.
- Route family: CHD-RM v5.
- Test policy: Haze4K locked test is final-confirmation only.

## Invariants

1. Start from `github/codex/haze4k-official-arch-anchor`.
2. Keep the route inside research content one: continuous haze-density-aware region-adaptive residual modulation with low-haze protection.
3. Do not turn the route into independent color, luminance, texture, or structure modeling.
4. Do not use Lab, luminance, gradient, or texture as core training targets.
5. Do not replace the ConvIR-B backbone in this route.
6. Do not connect or train RARM before density/need calibration, control, recall-protection, and no-op equivalence gates pass.
7. Do not use Haze4K locked test for checkpoint, threshold, route, scale, gamma, mask, loss, or hyperparameter selection.
8. Any candidate claim must beat matched-budget controls, not only A0.
9. Any final candidate must preserve low-haze regions and tail safety.

## Route Table

| Stage | Branch | Status | Main Result | Decision | Evidence Root |
| --- | --- | --- | --- | --- | --- |
| v0 route lock | `codex/haze4k-v5-v0-chd-rm-route-lock` | completed | route scope, locked-test policy, stages, and archive paths fixed | `COMPLETED_V0_ROUTE_LOCK` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v0_route_lock_20260708/` |
| v1 data baseline lock | `codex/haze4k-v5-v1-chd-rm-data-baseline-lock` | completed | train/test manifest, 2400/600 split, OOF folds, A0 val600, metric reproducibility, and efficiency locked | `COMPLETED_V1_GATE_PASS` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/` |
| v2 density-need calibration | `codex/haze4k-v5-v2-chd-rm-density-need-calibration` | paused | density passes strongly; need remains below gate; shuffled control fails | `PAUSE_V2_DUAL_HEAD_NOT_PASSED` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/` |
| v2b need calibration repair | `codex/haze4k-v5-v2b-chd-rm-need-calibration-repair` | paused | D6c improves need ranking but strong-response coverage remains 0 | `PAUSE_V2B_NEED_REPAIR_NOT_PASSED` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2b_need_calibration_repair_20260708/` |
| v2c need coverage calibration | `codex/haze4k-v5-v2c-chd-rm-need-coverage-calibration` | paused | Train-inner calibration restores coverage but creates unsafe false-strong responses | `PAUSE_V2C_SCALE_CALIBRATION_NOT_ENOUGH` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2c_need_coverage_calibration_20260709/` |
| v2d need spatial hard-negative | `codex/haze4k-v5-v2d-chd-rm-need-spatial-hard-negative` | paused | D7c frozen multi-context top-k HN is promising, but controls remained weak | `PAUSE_V2D_D7C_TOPK_PROMISING_BUT_CONTROLS_WEAK_NO_V3` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/` |
| v2e D7c control recall audit | `codex/haze4k-v5-v2e-chd-rm-d7c-control-recall-audit` | paused | Fixed permutation and density matched controls are clean, but D7c top-k LDHN recall is low and D7c-RP has no safe recall-protected point | `PAUSE_V2E_D7C_RP_NO_SAFE_RECALL_PROTECTED_POINT_NO_V3` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2e_d7c_control_recall_audit_20260709/` |
| v2f need target/head redesign | `codex/haze4k-v5-v2f-chd-rm-need-target-head-redesign` | paused | F0-F3 showed LDHN support and frozen-feature separability, but F4/F4b could not satisfy LDHN recall and false-tail safety together | `PAUSE_V2F_F4B_NO_SAFE_LDHN_POINT_NO_F5_NO_V3` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2f_need_target_head_redesign_20260709/`; `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2f_need_target_head_redesign_f4b_tail_rescue_20260709/` |
| v2g need actionability audit | `codex/haze4k-v5-v2g-chd-rm-need-actionability-audit` | paused | G1-G4a show global LDHN is over-broad and D7c beats deployable density controls under the three-state target; G4b selective probes did not safely improve over D7c | `PAUSE_G4B_SELECTIVE_PROBE_NO_SAFE_IMPROVEMENT_NO_F5_NO_V3` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2g_need_actionability_audit_20260709/` |
| v2h actionable prior sufficiency | `codex/haze4k-v5-v2h-actionable-prior-sufficiency` | completed with D preflight blocked | D7c A/B/C passed prior sufficiency; FAM2 no-op must move to a separate architecture branch | `V2H_ABC_PASS_D_BLOCKED_CREATE_SEPARATE_NOOP_ARCH_BRANCH` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2h_actionable_prior_sufficiency_20260709/` |
| v2i FAM2 no-op arch equivalence | `codex/haze4k-v5-v2i-fam2-noop-arch-equivalence` | completed | FAM2-only zero-init architecture insertion from official anchor is exact A0-equivalent on random input, real train-derived batch, and internal val-inner 600 | `V2I_FAM2_NOOP_ARCH_EQUIVALENCE_PASS_AUTHORIZE_D7C_GATED_NOOP_CONNECTION_ONLY` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2i_fam2_noop_arch_equivalence_20260710/` |
| v3a D7c-gated no-op connection audit | `codex/haze4k-v5-v3a-d7c-gated-noop-connection-audit` | completed no-training audit | D7c gate tensors are connected into FAM2 as an external gate tensor; final zero-init modulation remains exact A0-equivalent on random, real-batch, and internal val-inner 600 checks | `V3A_D7C_GATED_NOOP_CONNECTION_PASS_AUTHORIZE_NO_TRAINING_RARM_PREFLIGHT_ONLY` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v3a_d7c_gated_noop_connection_audit_20260710/` |
| v3b RARM preflight design | `codex/haze4k-v5-v3b-rarm-preflight-design` | completed preflight blocked | current train/valid/eval and modulation-stat entrypoints do not compute or pass the D7c gate required by `fam2_d7c_noop`; cloud v3a workspace is also dirty and not a clean parent runtime workspace | `V3B_RARM_PREFLIGHT_BLOCKED_GATE_PIPELINE_ABSENT_NO_RARM_TRAINING` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v3b_rarm_preflight_design_20260710/` |
| v3c gate forward contract | `codex/haze4k-v5-v3c-gate-forward-contract` | completed no-training preflight pass | D7c gate producer, partial A0 init, train/valid/eval forward helpers, and modulation-stat gate path passed on 16 internal val-inner samples with exact A0-equivalent outputs | `V3C_GATE_FORWARD_CONTRACT_PASS_AUTHORIZE_NO_TRAINING_ENTRYPOINT_PREFLIGHT_ONLY` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v3c_gate_forward_contract_20260710/` |
| v3d RARM adapter-only preflight | `codex/haze4k-v5-v3d-rarm-adapter-only-preflight` | paused after matched-control gate | Stage 0 exact no-op/freeze/gradient checks passed; D7c-gated 5-epoch adapter-only was safer than ungated control but did not beat matched-budget mean utility (`+0.02947 dB` vs control `+0.03307 dB`) | `V3D_PAUSE_D7C_SAFER_BUT_NOT_MATCHED_CONTROL_UTILITY_NO_20EPOCH_NO_V4` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710/` |
| v3e matched utility mechanism audit | `codex/haze4k-v5-v3e-matched-utility-mechanism-audit` | completed mechanism audit | Paired mean remains inconclusive but D7c tail safety is stable; 2x2 replay and gain audit show hard gate is a safety valve and D7c score is near-random for current FAM2 operator gain | `V3E_OPERATOR_CORRECTABILITY_MISMATCH_PRIMARY_HARD_GATE_SAFETY_TRADEOFF_SECONDARY_NO_RARM_EXPANSION` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v3e_matched_utility_mechanism_audit_20260710/` |
| v3f operator-correctability ranker audit | `codex/haze4k-v5-v3f-operator-correctability-ranker` | completed gate stop | D7c-vetoed gain oracle has useful upper-bound value, but best deployable scalar proxy AUROC is only `0.532034`, below the `0.56` training gate | `V3F_A_SCALAR_PROXY_SEPARABILITY_WEAK_NO_RANKER_TRAINING` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v3f_operator_correctability_ranker_20260710/` |
| v3m blockwise counterfactual advantage | `codex/haze4k-v5-v3m-blockwise-counterfactual-advantage` | completed gate stop | Block16 common-action oracle retained about 85% of pixel-grid lift and A1/A2 found strong direct-step-energy label observability, but A3 frozen policy replay retained only about 23% of block16 oracle lift and created severe tail regressions; corrected post-fail decomposition shows severe/hard failures are highly cross-operator stable and aggressive calibration bins over-escalate heavily | `V3M_A3_FROZEN_POLICY_REPLAY_FAIL_STOP_NO_ROUTE_CONFIRM`; diagnostic `V3M_A3_FAILURE_DECOMPOSITION_DIAGNOSTIC_ONLY_NO_AUTHORIZATION` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v3m_blockwise_counterfactual_advantage_20260711/` |
| v3n conservative first-step calibration | `codex/haze4k-v5-v3n-conservative-first-step-calibration` | completed gate stop | Fixed false-intervention semantics (`alpha=0.125` default, only `.25` escalation above 99th-percentile train-negative `direct_step_energy`) selected zero held-out blocks for both operators, so no replay is authorized | `V3N_A0_CONSERVATIVE_LABEL_PREFLIGHT_FAIL_STOP_NO_REPLAY` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v3n_conservative_first_step_calibration_20260712/` |
| v3o signed adjacent-advantage identifiability | `codex/haze4k-v5-v3o-signed-adjacent-advantage-identifiability` | A0 smoke gate passed | Fixed `alpha=.125` replay was exact on 32 OOF images per operator and block-SSE aggregation errors were below `1e-10`; formal 1,200-image candidate-SSE audit only is authorized | `V3O_A0_SMOKE_REPLAY_INTEGRITY_PASS_AUTHORIZE_FORMAL_OOF_ONLY` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v3o_signed_adjacent_advantage_identifiability_20260712/` |
| v3 no-op RARM audit | `codex/haze4k-v5-v3-chd-rm-noop-rarm-audit` | superseded by v3a naming | original v3 remains blocked as RARM route; use v3a for D7c-gated no-op connection only | `SUPERSEDED_BY_V3A_NOOP_CONNECTION_AUDIT` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v3_noop_rarm_audit_20260708/` |
| v4 single-scale RARM | `codex/haze4k-v5-v4-chd-rm-single-scale-rarm` | blocked | blocked until v3 no-op gate is authorized and passed | `BLOCKED` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v4_single_scale_rarm_20260708/` |
| v5 low-haze protection | `codex/haze4k-v5-v5-chd-rm-low-haze-protection` | blocked | blocked until a safe R_need/RARM gate exists | `BLOCKED` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v5_low_haze_protection_20260708/` |
| v6 multiscale haze modulation | `codex/haze4k-v5-v6-chd-rm-multiscale-haze-modulation` | blocked | blocked until earlier gates pass | `BLOCKED` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v6_multiscale_haze_modulation_20260708/` |
| v7 OOF candidate lock | `codex/haze4k-v5-v7-chd-rm-oof-candidate-lock` | blocked | blocked until candidate exists | `BLOCKED` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v7_oof_candidate_lock_20260708/` |
| v8 final Haze4K confirmation | `codex/haze4k-v5-v8-chd-rm-final-haze4k-confirmation` | blocked | blocked until v7 candidate lock | `BLOCKED` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v8_final_haze4k_confirmation_20260708/` |

## v2e Closeout

D7c top-k should be retained as evidence of real `R_need` ranking signal, not promoted to RARM. v2e fixed permutation and density-only matched controls are clean. The remaining blocker is safety/recall incompatibility:

- frozen D7c top-k LDHN recall `0.0370` with safe false-tail;
- first LDHN-passing RP point LDHN recall `0.1096` but false-p90 `0.0599` and false-p95 `0.2069`;
- strongest RP point LDHN recall `0.1822` but false-p95 `0.5348`.

Current next action is not v3/RARM. Any continuation must diagnose or redesign the frozen-side-head `R_need` target/head so LDHN recall and false-tail safety pass together.

## v2f First-Stage Decision

v2f F0-F3/F2 completed on `convir-4090` without D2, v3, RARM, ConvIR-B
unfreeze, or locked Haze4K test. First-stage evidence supports a bounded F4
canary:

- LDHN pixel coverage `0.08988972981770833`.
- LDHN core fraction of LDHN `0.569798970635499`.
- LDHN unstable fraction of LDHN `0.04701398288013833`.
- Best frozen feature probe `feature_set_2` + `mlp`: AUROC
  `0.8107264347671554`, AUPRC `0.807792756659645`.
- Density-conditioned target density Spearman `0.007215705298292346`, compared
  with global target density Spearman `0.31464418569286756`.

F4 density-stratified frozen-side `R_need` head canary then ran on
`train_inner`/`val_inner` and failed the original v2e global LDHN/false-tail
gate. The supplemental F4b tail-rescue matrix also failed:

- F4 selected variants had `safe_and_ldhn_points = 0`.
- F4b selected variants also had `safe_and_ldhn_points = 0`.
- Best F4b safe LDHN recall was only `0.0523`.
- F4b variants that reached high LDHN recall had false-p95 near `0.9895` to
  `1.0000`.

Decision: `PAUSE_V2F_F4B_NO_SAFE_LDHN_POINT_NO_F5_NO_V3`. Do not run F5,
v3, RARM, D2, ConvIR-B unfreeze, or locked Haze4K test from v2f. Do not repeat
F4/F4b strength sweeps without changing target semantics or available
information.

## v2g Actionability Audit Closeout

v2g tested whether the v2f failure was caused by the old global-LDHN target
being over-broad as a hard RARM-positive signal. It completed G0 source
reproduction, G1 semantic audit, G2/G2b available-information and oracle-gain
diagnostics, G3 three-state actionable target definition, G4a actionability
controls, and G4b selective-probe screening. No locked Haze4K test, D2, RARM,
v3, F5, or saved probe weights/checkpoints were used.

Key results:

- LDHN coverage is `0.089890`, but isolated LDHN fraction is `0.890713` and
  adjacent-to-haze fraction is only `0.109287`.
- Under the three-state target, D7c has action recall `0.548312`,
  low-adjacent recall `0.155904`, negative false rate `0.002974`, and isolated
  LDHN hit rate `0.022366`.
- D7c beats deployable D3 density control under that target: action recall
  `0.548312` vs `0.454247`, low-adjacent recall `0.155904` vs `0.113905`,
  negative false rate `0.002974` vs `0.049584`, and AUROC action-vs-negative
  `0.969589` vs `0.872087`.
- G4b selective probes did not beat D7c safely. The best probe,
  `context_image_density_linear`, had action recall `0.488995`, low-adjacent
  recall `0.076751`, negative false rate `0.004045`, and AUROC
  action-vs-negative `0.937536`. Versus D7c this is action recall `-0.059317`,
  low-adjacent recall `-0.079153`, negative false `+0.001071`, and AUROC
  `-0.032053`.

Decision: `PAUSE_G4B_SELECTIVE_PROBE_NO_SAFE_IMPROVEMENT_NO_F5_NO_V3`. This
does not authorize F5, v3, RARM, D2, adapter training, canary expansion, or
locked-test access. Do not repeat F4/F4b strength sweeps or a simple G4b
selective-probe rerun without a new written route decision.

## v2h A/B Closeout

v2h tested whether D7c is sufficient as a deployable actionable prior before any
future no-op/RARM/adapter work. It ran only risk-coverage and diagnostic
shadow-modulation audits on the internal Haze4K split. No locked test, D2, F5,
v3, RARM connection, RARM training, adapter training, new head family, or canary
expansion was run.

v2h-A fixed D7c operating point:

- coverage `0.302695`;
- action recall `0.548312`;
- low-adjacent recall `0.155904`;
- negative false rate `0.002974`;
- isolated-LDHN hit rate `0.022366`;
- per-image negative false p95 `0.047619`.

The density-matched control at comparable coverage was worse: action recall
`0.448391` and negative false rate `0.047786`.

v2h-B alpha `0.3` shadow-modulation diagnostic:

- D7c global PSNR gain `1.374164`;
- density-matched global PSNR gain `0.977430`;
- action-oracle global PSNR gain `2.220821`;
- D7c action-region gain `1.695614`;
- D7c negative touch `0.002698`;
- D7c isolated touch `0.023606`.

Decision: `V2H_AB_PASS_PRIOR_SUFFICIENT_AUTHORIZE_OOF_AND_NOOP_ONLY`. D7c is sufficient to justify v2h-C OOF stability and
v2h-D FAM2 no-op equivalence review only. The remaining bottleneck is connection
risk, not deployable-prior existence. RARM/training/locked-test access remain
blocked.

## v2h C/D Closeout

v2h-C ran the authorized no-training fold calibration stability audit over the
v1 fixed five-fold train OOF table. D7c stayed stable and safer than density
matching:

- D7c action recall mean/min `0.576335` / `0.556955`;
- D7c low-adjacent recall mean `0.170063`;
- D7c negative false mean/max `0.003403` / `0.003996`;
- D7c selected coverage std `0.010785`;
- density-matched negative false mean/max `0.049636` / `0.063885`.

v2h-D attempted the authorized FAM2/no-op equivalence review but stopped at the
correct preflight boundary: the v2h branch preserves the official architecture
anchor and rejects `fam2_modres`. This means no-op insertion must be designed on
a separate model-structure branch from `github/codex/haze4k-official-arch-anchor`.
It does not authorize RARM/training/locked-test access.

Decision: `V2H_ABC_PASS_D_BLOCKED_CREATE_SEPARATE_NOOP_ARCH_BRANCH`.

## v2i Route Start

v2i is the separate model-structure no-op audit required by v2h-D. It starts
from `github/codex/haze4k-official-arch-anchor` and inserts only a FAM2
zero-init modulation shell:

- FAM1 remains original.
- Candidate mode is only `fam2_modres`.
- Expected new keys are exactly `FAM2.modulator.weight` and
  `FAM2.modulator.bias`.
- Expected parameter delta is `8320`.
- Random input, real Haze4K train-derived batch, and internal val-inner 600 final
  outputs must be no-op equivalent to A0 with max absolute difference `<= 1e-7`.
- PSNR/SSIM deltas must be numerically equivalent.
- No training, RARM connection, D7c forward injection, adapter training, or
  locked Haze4K test is authorized.

v2i passed on `convir-4090`:

- candidate missing keys exactly `FAM2.modulator.weight` and
  `FAM2.modulator.bias`;
- unexpected keys and shape mismatches empty;
- parameter delta exactly `8320`;
- FAM2 modulator weight/bias stats all zero;
- random tensor max/mean abs diff `0.0` / `0.0`;
- real train-derived batch max/mean abs diff `0.0` / `0.0`;
- internal val-inner 600 max abs diff `0.0`;
- internal val-inner 600 PSNR/SSIM max absolute deltas `0.0` / `0.0`;
- no training, RARM, D7c forward connection, adapter training, or locked test.

Decision: `V2I_FAM2_NOOP_ARCH_EQUIVALENCE_PASS_AUTHORIZE_D7C_GATED_NOOP_CONNECTION_ONLY`.

This authorizes only a separate D7c-gated no-op connection audit, not training.

## v3a Closeout

v3a ran the authorized D7c-gated FAM2 no-op connection audit on `convir-4090`.
The route starts from `github/codex/haze4k-official-arch-anchor`, connects D7c
gate tensors into FAM2 as an external gate tensor, and keeps final gamma/beta
modulation zero-initialized.

v3a passed:

- candidate missing keys exactly `FAM2.modulator.weight` and
  `FAM2.modulator.bias`;
- unexpected keys and shape mismatches empty;
- parameter delta exactly `8320`;
- random and real-batch no-op equivalence passed;
- internal val-inner 600 output max absolute diff `0.0`;
- internal val-inner 600 PSNR/SSIM max absolute deltas `0.0` / `0.0`;
- nontrivial D7c gate coverage `599/600`;
- no locked Haze4K test, no training, no RARM, no adapter training, and no
  ConvIR-B unfreeze.

Attempts 1-4 were engineering/audit closeout issues, not scientific failures:
CUDA no-op expression perturbation, deterministic audit setup, obsolete
modulator shape expectation, and a missing-key order comparison bug. Attempt 5
is the final valid pass.

Decision:
`V3A_D7C_GATED_NOOP_CONNECTION_PASS_AUTHORIZE_NO_TRAINING_RARM_PREFLIGHT_ONLY`.

This pass authorizes only a separate preflight/design decision for any next
route. It does not authorize RARM, training, adapter work, canary expansion, or
locked-test access.

## v3b Closeout

v3b completed the separate no-training preflight/design decision authorized by
v3a and stopped before any RARM or training launch. The preflight question was
whether current runnable entrypoints can support `fam2_d7c_noop` without a new
gate-producing runtime design.

The answer is no:

- `fam2_d7c_noop` raises if `d7c_gate` is absent in both forward and modulation
  stats paths;
- `train.py` still calls `pred_img = model(input_img)`;
- `valid.py` and `eval.py` still call `model(input_img)[2]`;
- train-time modulation stats still call
  `model.collect_modulation_stats(input_img)`;
- the existing cloud v3a workspace is dirty and behind the GitHub v3a pass
  commit, so it is not a clean parent runtime workspace.

Decision:
`V3B_RARM_PREFLIGHT_BLOCKED_GATE_PIPELINE_ABSENT_NO_RARM_TRAINING`.

This is not a v3a no-op numerical failure. It is an entrypoint/contract
blocker: v3a proved that an externally supplied D7c gate can be no-op connected,
but no approved train/valid/eval gate-producing pipeline exists yet. Do not run
RARM, training, adapter work, canary expansion, or locked-test access from v3b.
Any future continuation must first write and audit a gate-producing forward
contract as a no-training preflight.

## v3c Closeout

v3c implemented and audited the gate-producing forward contract required by
v3b. The route branch adds a frozen D7c gate producer, partial A0 init support
for the zero-init FAM2 modulator keys, and train/valid/eval/modulation-stat
helpers that pass `d7c_gate` into `fam2_d7c_noop`.

The cloud no-training audit passed on `convir-4090` from a fresh workspace at
route commit `0a350393776c4263386c72c8b81be076d9d984a5`:

- source contract checks passed;
- official A0 partial init missed exactly `FAM2.modulator.weight` and
  `FAM2.modulator.bias`;
- checked samples: `16` internal val-inner images;
- nontrivial D7c gate images: `16/16`;
- D7c selected coverage mean/min/max:
  `0.3246304675703868` / `0.015908146277070045` /
  `0.6701125502586365`;
- output max absolute diff: `0.0`;
- PSNR/SSIM max absolute deltas: `0.0` / `0.0`;
- modulation stats include D7c gate stats;
- no training, RARM, adapter training, ConvIR-B unfreeze, canary expansion, or
  locked Haze4K test was used.

Decision:
`V3C_GATE_FORWARD_CONTRACT_PASS_AUTHORIZE_NO_TRAINING_ENTRYPOINT_PREFLIGHT_ONLY`.

This resolves the v3b entrypoint-contract blocker only. It still does not
authorize RARM or training. Any next RARM/training step needs its own written
decision, resource preflight, metric contract, and stage gate.


## v3e Closeout

v3e ran only no-training mechanism audits on `convir-4090`; no locked test,
training continuation, checkpoint-producing run, v4/RARM expansion,
neighbor/FAM1/backbone unfreeze, or new generic D7c probe was used.

Decision:
`V3E_OPERATOR_CORRECTABILITY_MISMATCH_PRIMARY_HARD_GATE_SAFETY_TRADEOFF_SECONDARY_NO_RARM_EXPANSION`.

Key evidence:

- v3e-A paired bootstrap: D7c-control mean CI95 `[-0.01676, -0.00365, +0.00930]`,
  so single-seed mean ordering remains inconclusive; `<= -0.2 dB` tail-regression
  reduction CI95 `[26, 41, 57]`, so D7c tail safety is stable.
- v3e-B 2x2 replay: `W_D+G_D` mean `+0.02947` with `50` regressions;
  `W_D+G_1` mean `+0.03899` with `113` regressions; `W_U+G_D` mean `+0.01278`
  with `23` regressions; `W_U+G_1` mean `+0.03307` with `91` regressions. The
  hard gate is a real safety valve but drops ungated mean utility.
- v3e-C operator-gain audit: D7c score vs ungated FAM2 positive gain AUROC
  `0.4921`; D7c score vs D7c-gated FAM2 positive gain AUROC `0.4904`. D7c
  actionability is not a current-FAM2 action-value router.
- v3e-D training-contract audit: all audited batches were clipped hard; effective
  Adam weight decay was `0` despite CLI `0.0001`; resume checkpoints contain no
  scheduler state. Component gradient cosines were positive in this small audit,
  so loss-gradient conflict is not the primary proven blocker.

This authorized the separate v3f design/audit route for `D7c safety veto + FAM2
operator-correctability ranker` using internal/OOF actual FAM2 marginal gain
targets only. The v3f closeout below supersedes this as the current status. Do
not continue v3d or silently fix optimizer/scheduler and compare directly to
v3d.

## v3f Closeout

v3f ran the authorized no-training correctability target/separability audit on
`convir-4090`. It used internal train-derived val-inner 600 only, sampled
`4,915,200` pixels for target/proxy metrics, produced no checkpoints, and did
not touch the Haze4K locked test.

Decision:
`V3F_A_SCALAR_PROXY_SEPARABILITY_WEAK_NO_RANKER_TRAINING`.

Key evidence:

- best deployable scalar proxy: FAM2 correction magnitude, positive-gain AUROC
  `0.532034`, AUPRC `0.519730`, gain Spearman `0.033662`;
- D7c score and hard gate positive-gain AUROC: `0.492237` and `0.492251`;
- ungated control `W_U+G_1`: mean PSNR delta `+0.033065`, `91`
  `<= -0.2 dB` regressions;
- D7c-vetoed control `W_U+G_D`: mean PSNR delta `+0.012366`, `18`
  `<= -0.2 dB` regressions;
- D7c-vetoed gain oracle: mean PSNR delta `+0.078254`, zero `<= -0.2 dB`
  regressions.

The oracle result confirms correctability exists in principle inside the D7c
safety region, but the audited deployable scalar features are too weak to
recover it. Therefore v3f-B ranker training is not authorized. Do not launch
v3d continuation, 20-epoch runs, v4/RARM expansion, neighbor/FAM1/backbone
unfreeze, canary expansion, or locked-test access from this evidence. Any future
route must introduce new operator-context features, operator target semantics,
or a different correction operator before another correctability-ranker screen.

## Gate Summary

| Stage | Must Pass Before |
| --- | --- |
| v1 data/baseline lock | any density/need training |
| v2 density/need calibration | RARM connection |
| v2e control and recall audit | v2f target/head redesign only |
| v2f target/head redesign | v2g actionability/prior diagnostics only; no F5, v3, RARM, D2, or locked test |
| v2g target actionability audit | no-training prior-sufficiency/no-op architecture audits only; no F5, v3, RARM, D2, or locked test |
| v2h actionable prior sufficiency | separate FAM2 no-op architecture branch only |
| v2i FAM2 no-op arch equivalence | D7c-gated no-op connection audit |
| v3a D7c-gated no-op connection audit | a separate preflight/design decision only |
| v3b RARM preflight design | gate-producing train/valid/eval forward contract; no RARM training |
| v3c gate forward contract | separate written RARM/training decision only |
| v3d matched-control utility gate | v3e mechanism audit only; no 20-epoch, v4, neighbor/FAM1/backbone unfreeze, or locked test |
| v3e matched utility mechanism audit | separate v3f design/audit for D7c safety veto plus FAM2 operator-correctability ranker only; no direct RARM expansion |
| v3f operator-correctability ranker audit | no v3f-B scalar-feature ranker training; future work needs new operator context, target semantics, or correction operator |
| v3 no-op RARM audit | RARM training |
| v4 single-scale matched controls | final candidate consideration |
| v5 low-haze protection | final candidate consideration |
| v6 multiscale check | selecting CHD-RM-MS over CHD-RM-LP |
| v7 OOF candidate lock | any Haze4K locked-test command |
| v8 final confirmation | promotion wording |

## Metric Families

- restoration quality: PSNR, SSIM, LPIPS, dPSNR, dSSIM, dLPIPS;
- region quality: low/medium/heavy/very-heavy haze PSNR and dPSNR;
- statistics: mean, median, positive ratio, p5, p10, CVaR5, worst32, bootstrap CI, sign-test p;
- calibration: density/need Pearson, Spearman, AUROC, AUPRC, monotonicity, mask coverage, false strong-recovery rate;
- modulation: gamma means by bucket, gamma correlations, residual norms;
- efficiency: params, FLOPs, FPS, latency, peak GPU memory, training time.

## Pause Rules

- Pause immediately if a required asset, split, checkpoint, or Python path is missing.
- Pause if a command tries to touch locked test before v7 candidate lock.
- Pause if local-only execution would be needed for runtime validation.
- Pause if a stage gate fails and the next step would expand scope instead of diagnosing the failed gate.
