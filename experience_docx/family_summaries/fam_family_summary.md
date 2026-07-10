# FAM/FAM2 Family Summary

Date: 2026-07-10

Status: closed for unchanged deployable FAM routing; reopened only for
no-training operator-site context audits after v3g.

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
- Evidence roots:
  - `../experiment_logs/haze4k_fam_modres_scout_stop5_20260531/`
  - `../experiment_logs/haze4k_fam2_modres_stop20_20260531/`
  - `../experiment_logs/haze4k_fam2_bounded_gamma_stop20_20260601/`
  - `../experiment_logs/haze4k_fam2_conf_gate_stop20_20260601/`
  - `../experiment_logs/haze4k_fam2_selectivity_or_kill_20260601/`
  - `../experiment_logs/haze4k_v5_chd_rm_v3g_fam2_action_space_correctability_20260710/`

## Established Facts

| Route | Main result | Decision |
| --- | --- | --- |
| FAM `modres` stop5 | Mean PSNR `+0.0953 dB`, but median delta negative and strong-reference regressions `142/250`. | Do not promote unchanged `modres`; mechanism active but preservation fails. |
| FAM2-only stop20 | Mean `+0.1739 dB`; hard bottom-25% `+0.8159 dB`; easy top-25% `-0.2860 dB`; strong-reference regressions `138/250`. | Diagnostic only; preservation gate fails. |
| FAM2 bounded gamma | Mean `-0.0271 dB`; hard `+0.8054 dB`; easy `-1.2740 dB`; strong-reference regressions `181/250`. | Bounded gamma does not solve preservation. |
| FAM2 confidence-gated gamma | Mean `+0.4523 dB`; hard `+0.9380 dB`; easy `-0.0700 dB`; strong-reference regressions `121/250`. | Positive quality signal, but preservation/selectivity not decision-grade. |
| FAM2 selectivity-or-kill | Deployable selectors passing gate: `0`; best positive-gain AUC `0.5874`; best feasible threshold-gate mean `+0.1333 dB`. | `FAIL_STOP_FAM_ROUTE`. |
| CHD-RM v3g FAM2 action-space audit | Label-derived FAM2 alpha oracle reached mean `+0.412676 dB`, median `+0.338481 dB`, p10 `+0.060075 dB`; finite-difference sign agreement `0.910641`. Hard D7c and ungated action replay remained weak/tail-risky. | `V3G_ACTION_ORACLE_STRONG_FEATURES_WEAK_REQUIRE_OPERATOR_CONTEXT_NO_TRAINING`. |

## Family Verdict

FAM/FAM2 established that hard Haze4K samples can be moved, but the tested
feature modulation and gamma variants repeatedly harmed easy or already-strong
reference cases. The original no-training selectivity analysis did not find a
deployable selector strong enough to safely decide when to apply the FAM
intervention.

v3g refines, rather than reverses, that verdict: the FAM2 actuator itself has a
strong action-space oracle when alpha is chosen with label-derived gate-site
gradients, but current deployable routing signals are still insufficient. The
current blocker is deployable operator-site context/controller quality.

The family remains closed for unchanged deployable FAM routing and for direct
router/ranker training from scalar or image-level proxies. It may reopen only
for no-training operator-site context audits that test inference-time features
at the true FAM2 action site.

## Do Not Repeat Without New Evidence

- Do not rerun unchanged FAM/FAM2 modulation just because hard-bucket gains were
  large; the preservation failure is already documented.
- Do not treat average PSNR gains as sufficient when strong-reference
  regressions remain near `121/250` or worse.
- Do not launch another FAM selector unless its preflight shows stronger
  held-out separability than the failed selectivity-or-kill analysis.
- Do not train v3f-B or a scalar-feature ranker from v3g; v3g authorizes only
  no-training operator-site context feature audits.
- Do not continue the same v3d FAM2 adapter, launch 20-epoch continuation, use
  locked Haze4K test, or expand to v4/RARM from this evidence.

## Reopen Condition

A FAM-family route can reopen for training only if a no-training operator-site
context audit first shows that inference-time features at or near the true FAM2
action site can recover the v3g action target with a predeclared held-out
separability and replay gate, including explicit false-intervention protection
for strong-reference/easy images.
