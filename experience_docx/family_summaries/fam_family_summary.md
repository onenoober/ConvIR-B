# FAM/FAM2 Family Summary

Date: 2026-07-11

Status: closed for unchanged deployable FAM routing, current v3h/v3i signal
sets, and v3j tiny direct bounded residual heads after v3j.

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
OOF and route-confirm. The family remains closed for unchanged deployable FAM
routing, direct router/ranker/distillation from the current signal sets, tiny
direct bounded residual heads, and continued sweeping of the same
operator/context features. It may reopen only with materially new tail-risk
information, target semantics, a joint correction-confidence design, or bounded
experts that first pass a held-out replay gate with explicit false-intervention
protection.

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
  counterfactual-response, or disagreement signal sets; they already failed
  held-out no-training OOF/replay audits.
- Do not treat positive mean gains from v3j-style direct residual heads as
  sufficient; v3j-B improved mean but produced `121/600` route-confirm severe
  regressions for both direct heads.

## Reopen Condition

A FAM-family route can reopen for training only if materially new tail-risk
information, target semantics, a different controller source, joint
correction-confidence design, or bounded experts first pass a held-out
separability/replay gate, including explicit false-intervention protection for
strong-reference/easy images. Reusing the current v3h/v3i signal sets or the
v3j tiny direct-residual formulation is not sufficient.
