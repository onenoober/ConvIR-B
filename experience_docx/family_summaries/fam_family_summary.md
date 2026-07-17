# FAM/FAM2 Family Summary

Date: 2026-07-17

Status: closed for unchanged deployable FAM routing, current v3h/v3i signal
sets, v3j tiny direct bounded residual heads, v3k provisional micro-alpha, v3l
transmission-only physics risk, v3m direct-step-energy block policy replay,
v3n conservative first-step direct-step-energy thresholding, and v3p scalar-A
physics on the current Haze4K package.

The 2026-07-17 R3 evidence audit does not reopen unchanged FAM/FAM2 routing. It
clarifies that v3i-C closes a single fixed-action counterfactual RGB-response
probe, not every multi-action action-conditioned value model.

## Sources

- Index: `../EXPERIMENT_INDEX.md`
- CHD-RM index: `../CHD_RM_EXPERIMENT_INDEX.md`
- Cards:
  - `../experiment_cards/2026-05-31-haze4k-fam-feature-modulation.md`
  - `../experiment_cards/2026-05-31-haze4k-fam2-only-modulation.md`
  - `../experiment_cards/2026-06-01-haze4k-fam2-bounded-modulation.md`
  - `../experiment_cards/2026-06-01-haze4k-fam2-confidence-gate.md`
  - `../experiment_cards/2026-06-01-haze4k-fam2-selectivity-or-kill.md`
  - `../experiment_cards/haze4k-chd-rm-v3g-fam2-action-space-correctability.md`
  - `../experiment_cards/haze4k-chd-rm-v3h-operator-site-context-audit.md`
  - `../experiment_cards/haze4k-chd-rm-v3i-fam2-open-value-distillability.md`
  - `../experiment_cards/haze4k-chd-rm-v3j-bounded-safe-correction-audit.md`
  - `../experiment_cards/2026-07-11-haze4k-v5-chd-rm-v3k-tail-risk-observability.md`
  - `../experiment_cards/2026-07-11-haze4k-v5-chd-rm-v3l-safe-step-escalation-physics-audit.md`
  - `../experiment_cards/2026-07-11-haze4k-v5-chd-rm-v3m-blockwise-counterfactual-advantage.md`
  - `../experiment_cards/2026-07-12-haze4k-v5-chd-rm-v3n-conservative-first-step-calibration.md`
  - `../experiment_cards/2026-07-12-haze4k-v5-chd-rm-v3p-canonical-signed-gain.md`
  - `../experiment_cards/2026-07-17-haze4k-v5-r3-cloud-evidence-audit.md`
  - `../experiment_cards/2026-07-17-haze4k-v5-r3-proposal-first-acv-design.md`
- Evidence roots:
  - `../experiment_logs/haze4k_fam_modres_scout_stop5_20260531/`
  - `../experiment_logs/haze4k_fam2_modres_stop20_20260531/`
  - `../experiment_logs/haze4k_fam2_bounded_gamma_stop20_20260601/`
  - `../experiment_logs/haze4k_fam2_conf_gate_stop20_20260601/`
  - `../experiment_logs/haze4k_fam2_selectivity_or_kill_20260601/`
  - `../experiment_logs/haze4k_v5_chd_rm_v3g_fam2_action_space_correctability_20260710/`
  - `../experiment_logs/haze4k_v5_chd_rm_v3h_operator_site_context_audit_20260710/`
  - `../experiment_logs/haze4k_v5_chd_rm_v3i_fam2_open_value_distillability_20260711/`
  - `../experiment_logs/haze4k_v5_chd_rm_v3j_bounded_safe_correction_audit_20260711/`
  - `../experiment_logs/haze4k_v5_chd_rm_v3k_tail_risk_observability_20260711/`
  - `../experiment_logs/haze4k_v5_chd_rm_v3l_safe_step_escalation_physics_audit_20260711/`
  - `../experiment_logs/haze4k_v5_chd_rm_v3m_blockwise_counterfactual_advantage_20260711/`
  - `../experiment_logs/haze4k_v5_chd_rm_v3n_conservative_first_step_calibration_20260712/`
  - `../experiment_logs/haze4k_v5_chd_rm_v3p_canonical_signed_gain_20260712/`
  - `../experiment_logs/haze4k_v5_r3_cloud_evidence_audit_20260717/`

## Established Facts

| Route | Main result | Decision |
| --- | --- | --- |
| FAM `modres` stop5 | Mean PSNR `+0.0953 dB`, but median delta negative and strong-reference regressions `142/250`. | Do not promote unchanged `modres`; mechanism active but preservation fails. |
| FAM2-only stop20 | Mean `+0.1739 dB`; hard bottom-25% `+0.8159 dB`; easy top-25% `-0.2860 dB`; strong-reference regressions `138/250`. | Diagnostic only; preservation gate fails. |
| FAM2 bounded gamma | Mean `-0.0271 dB`; hard `+0.8054 dB`; easy `-1.2740 dB`; strong-reference regressions `181/250`. | Bounded gamma does not solve preservation. |
| FAM2 confidence-gated gamma | Mean `+0.4523 dB`; hard `+0.9380 dB`; easy `-0.0700 dB`; strong-reference regressions `121/250`. | Positive quality signal, but preservation/selectivity not decision-grade. |
| FAM2 selectivity-or-kill | Deployable selectors passing gate: `0`; best positive-gain AUC `0.5874`; best feasible threshold-gate mean `+0.1333 dB`. | `FAIL_STOP_FAM_ROUTE`. |
| CHD-RM v3g FAM2 action-space audit | Label-derived FAM2 alpha oracle reached mean `+0.412676 dB`, median `+0.338481 dB`, p10 `+0.060075 dB`; finite-difference sign agreement `0.910641`. Hard D7c and ungated action replay remained weak/tail-risky. | `V3G_ACTION_ORACLE_STRONG_FEATURES_WEAK_REQUIRE_OPERATOR_CONTEXT_NO_TRAINING`. |
| CHD-RM v3h operator-site context audit | Best holdout feature keep dir AUROC was only `0.504729`; best feature replay mean `+0.008995 dB` did not beat hard D7c action mean `+0.009352 dB`, while the oracle reference stayed `+0.420325 dB`. | `V3H_OPERATOR_CONTEXT_FEATURES_WEAK_NO_ROUTER_TRAINING`. |
| CHD-RM v3i FAM2 open-value distillability | Open-value oracle and compressed policies were strong (`ALPHA_SECANT_Q3` mean `+0.412879 dB`, zero severe), but full-context, counterfactual-response, and disagreement OOF probes all failed to stably beat hard D7c. | `V3I_ALL_DEPLOYABLE_SIGNALS_FAIL_STOP_FAM2_ROUTER_REDESIGN_CANDIDATE`. |
| CHD-RM v3j bounded safe-correction audit | Primary bounded output-residual teacher projection was safe (`PRIMARY_FULL_CLIP_P99_D7C` mean `+0.229641 dB`, p10 `+0.002454 dB`, zero severe), but deployable direct linear/context heads caused `121` route-confirm severe regressions each despite positive mean gains. | `V3J_DIRECT_SAFE_CORRECTION_OOF_FAIL_REQUIRE_NEW_INFORMATION_NO_INTERNAL_ROUTER`. |
| CHD-RM v3k tail-risk observability | Corrected the harmful full-step boundary to `alpha* = 0.5` and showed direct heads mix wrong-direction with harmful overshoot. `context alpha=0.125` was tail-safe on grouped OOF and historical open holdout, but reconstruction mismatch and missing new sealed split made the conclusion provisional. | `V3K_PROVISIONAL_MICRO_ALPHA_SAFE_STEP_SUPPORTED_NO_CANARY_NO_NEW_SEALED_SPLIT`. |
| CHD-RM v3l safe-step escalation and physics audit | Frozen context operators replayed exactly and oracle image/block/pixel step-size policies had large zero-severe upside, but privileged transmission-only features failed the direct-severe OOF AUC gate (`~0.635`/`~0.631` vs `0.65`). | `V3L_B_PRIVILEGED_TRANSMISSION_RISK_WEAK_STOP_NO_PHYSICS_POLICY`. |
| CHD-RM v3m blockwise counterfactual advantage | Common-action block16 oracle value and direct-step-energy label observability are real, but actual frozen policy replay retained only about `23%` of block16 oracle lift and created unsafe tails. Corrected post-fail decomposition shows severe/hard failures are highly stable across the two frozen operators and aggressive A2 bins over-escalate heavily. | `V3M_A3_FROZEN_POLICY_REPLAY_FAIL_STOP_NO_ROUTE_CONFIRM`; diagnostic `V3M_A3_FAILURE_DECOMPOSITION_DIAGNOSTIC_ONLY_NO_AUTHORIZATION`. |
| CHD-RM v3n conservative first-step calibration | A stricter false-intervention rule defaulting to `alpha=0.125` and allowing only `.25` above the 99th-percentile train-negative `direct_step_energy` threshold selected zero held-out blocks for both operators. | `V3N_A0_CONSERVATIVE_LABEL_PREFLIGHT_FAIL_STOP_NO_REPLAY`. |
| CHD-RM v3p canonical reconstruction and physics audit | Fresh float64 canonical reconstruction passes, and a fixed 25% constrained G1 oracle has nontrivial two-operator headroom. The scalar-A physics smoke has correct triplets but fails sRGB forward reconstruction far above its `8/255` margin; direct transmission semantic checks are worse. | `V3P_B0_SCALAR_A_SMOKE_FAIL_STOP_PHYSICS_ROUTE`; no B0 formal/B1 or physics estimator. |
| 2026-07-17 R3 cross-family audit | v3i-C's fixed-action response probe remains negative: best OOF mean `+0.008543 dB` and paired delta versus hard D7c CI95 low `-0.009492 dB`. It did not test explicit action identity, same-image multi-candidate ranking, regret weighting, or abstention. | Do not repeat fixed-action Y1-Y0. A separately registered proposal-first multi-action critic remains an open CHD-RM/R3 question, not a FAM2 route continuation. |

## Family Verdict

FAM/FAM2 established that hard Haze4K samples can be moved, but the tested
feature modulation and gamma variants repeatedly harmed easy or already-strong
reference cases. The original no-training selectivity analysis did not find a
deployable selector strong enough to safely decide when to apply the FAM
intervention.

v3g refined, rather than reversed, that verdict: the FAM2 actuator itself has a
strong action-space oracle when alpha is chosen with label-derived gate-site
gradients. v3h then tested the obvious deployable operator-site context features
and simple top/bottom replay policies; those features were near-random for the
action target and did not beat hard D7c mean utility on holdout.

v3i closed the remaining scalar/router question more directly. The privileged
open-value target is strong and spatially compressible, but full-context,
counterfactual-response, and checkpoint/transform disagreement signals all
failed OOF replay. v3j then removed the old FAM2 correction from the center of
the question: a bounded output-residual actuator is safe under the privileged
teacher, but deployable tiny direct residual heads create unsafe tails on both
OOF and route-confirm. v3k/v3l sharpened that into a step-size observability
problem: a tiny `alpha=0.125` step is tail-safer, and oracle step selection has
large zero-severe upside, but raw transmission metadata cannot identify
direct-severe risk strongly enough to become a policy. v3m then showed that
common-action block16 oracle value and direct-step-energy label observability
are not sufficient: fold-separated label calibration produced positive mean
PSNR but unsafe image-level tails, with corrected post-fail diagnostics showing
stable cross-operator severe/hard failures and heavy over-escalation in
aggressive calibration bins. v3n then tested the simplest conservative
first-step protection; it achieved zero false intervention only by selecting no
blocks, so `direct_step_energy` alone still does not provide an actionable safe
policy.

v3p changes the evidence classification without reopening the stopped policy:
the v3o numeric fail remains historical, but canonical float64 measurement and
the constrained G1 oracle are valid. This removes renderer and action-ladder
granularity as the immediate explanation, while preserving selection and
image-level harm as the bottleneck. The current Haze4K `haze/gt/trans` package
does not satisfy the scalar-A forward contract even under its directly specified
alternative semantics, so privileged physics cannot bridge that bottleneck.

The R3 audit narrows the v3i conclusion. The negative result applies to the
frozen single-action counterfactual response and its tested OOF compression.
It does not establish that response is universally uninformative when the model
is conditioned on explicit action identity and compares multiple candidates
within the same image. Such a test must live in the new proposal-first R3
factorial, include action/response shuffles and direct regret targets, and may
not reuse v3i outcomes as confirmation.

The family remains closed for unchanged deployable FAM routing, direct
router/ranker/distillation from the current signal sets, tiny direct bounded
residual heads, raw-transmission-only physics risk policies, and continued
sweeping of the same operator/context features. It may reopen only with
materially new tail-risk information, target semantics, a joint
correction-confidence design, or bounded experts that first pass clean OOF plus
new sealed-split replay gates with explicit false-intervention protection.

## Do Not Repeat Without New Evidence

- Do not rerun unchanged FAM/FAM2 modulation just because hard-bucket gains were
  large; the preservation failure is already documented.
- Do not treat average PSNR gains as sufficient when strong-reference
  regressions remain near `121/250` or worse.
- Do not launch another FAM selector unless its preflight shows stronger
  held-out separability than the failed selectivity-or-kill analysis.
- Do not train v3f-B or a scalar-feature ranker from v3g/v3h/v3i; v3i
  found the audited deployable signal families insufficient.
- Do not continue the same v3d FAM2 adapter, launch 20-epoch continuation, use
  locked Haze4K test, or expand to v4/RARM from this evidence.
- Do not keep sweeping the current v3h/v3i scalar, operator-site,
  fixed-single-action counterfactual-response, or disagreement signal sets;
  they already failed held-out no-training OOF/replay audits. A multi-action
  action-conditioned critic is materially different only when candidate
  identity, same-image ranking/regret, response shuffle, and abstention are all
  predeclared.
- Do not treat positive mean gains from v3j-style direct residual heads as
  sufficient; v3j-B improved mean but produced `121/600` route-confirm severe
  regressions for both direct heads.
- Do not treat `context alpha=0.125` as canary-authorizing without a new sealed
  validation split and deterministic saved operator artifacts.
- Do not build a physics-risk policy from raw Haze4K transmission alone; v3l
  privileged transmission failed the direct-severe OOF AUC gate.
- Do not treat v3m A1/A2 label observability as policy safety; v3m A3 replay
  failed tail gates and the corrected decomposition shows aggressive
  calibration bins mix oracle actions too heavily.
- Do not replay a conservative direct-step-energy `.125 -> .25` policy from
  v3n; the preregistered 99th-percentile negative threshold had zero held-out
  coverage.
- Do not relax v3p B0's `8/255` forward-residual margin, rerun B0 formal/B1,
  estimate `t_hat/A_hat`, or treat the current `trans` PNGs as a physics policy
  basis. The compact B0 diagnosis already tests raw/inverted transmission,
  channel-wise A, and filename-implied exponential alternatives.

## Reopen Condition

A FAM-family route can reopen for training only if materially new tail-risk
information, target semantics, a different controller source, joint
correction-confidence design, or bounded experts first pass clean OOF plus new
sealed-split separability/replay gates, including explicit false-intervention
protection for strong-reference/easy images. Reusing the current v3h/v3i signal
sets, the v3j tiny direct-residual formulation, v3l raw-transmission-only risk
features, v3m direct-step-energy mean-action calibration, or v3n conservative
direct-step-energy first-step thresholding is not sufficient.
The current Haze4K package can support a physics route only after independent
authoritative generator/serialization provenance validates a new data contract;
otherwise the next route must be a separately authorized candidate-pair value
assessor with frozen image-level harm gates.
