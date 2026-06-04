# FAM/FAM2 Family Summary

Date: 2026-06-04

Status: closed for unchanged deployable FAM routing.

## Sources

- Index: `../EXPERIMENT_INDEX.md`
- Cards:
  - `../experiment_cards/2026-05-31-haze4k-fam-feature-modulation.md`
  - `../experiment_cards/2026-05-31-haze4k-fam2-only-modulation.md`
  - `../experiment_cards/2026-06-01-haze4k-fam2-bounded-modulation.md`
  - `../experiment_cards/2026-06-01-haze4k-fam2-confidence-gate.md`
  - `../experiment_cards/2026-06-01-haze4k-fam2-selectivity-or-kill.md`
- Evidence roots:
  - `../experiment_logs/haze4k_fam_modres_scout_stop5_20260531/`
  - `../experiment_logs/haze4k_fam2_modres_stop20_20260531/`
  - `../experiment_logs/haze4k_fam2_bounded_gamma_stop20_20260601/`
  - `../experiment_logs/haze4k_fam2_conf_gate_stop20_20260601/`
  - `../experiment_logs/haze4k_fam2_selectivity_or_kill_20260601/`

## Established Facts

| Route | Main result | Decision |
| --- | --- | --- |
| FAM `modres` stop5 | Mean PSNR `+0.0953 dB`, but median delta negative and strong-reference regressions `142/250`. | Do not promote unchanged `modres`; mechanism active but preservation fails. |
| FAM2-only stop20 | Mean `+0.1739 dB`; hard bottom-25% `+0.8159 dB`; easy top-25% `-0.2860 dB`; strong-reference regressions `138/250`. | Diagnostic only; preservation gate fails. |
| FAM2 bounded gamma | Mean `-0.0271 dB`; hard `+0.8054 dB`; easy `-1.2740 dB`; strong-reference regressions `181/250`. | Bounded gamma does not solve preservation. |
| FAM2 confidence-gated gamma | Mean `+0.4523 dB`; hard `+0.9380 dB`; easy `-0.0700 dB`; strong-reference regressions `121/250`. | Positive quality signal, but preservation/selectivity not decision-grade. |
| FAM2 selectivity-or-kill | Deployable selectors passing gate: `0`; best positive-gain AUC `0.5874`; best feasible threshold-gate mean `+0.1333 dB`. | `FAIL_STOP_FAM_ROUTE`. |

## Family Verdict

FAM/FAM2 established that hard Haze4K samples can be moved, but the tested
feature modulation and gamma variants repeatedly harmed easy or already-strong
reference cases. The no-training selectivity analysis did not find a deployable
selector strong enough to safely decide when to apply the FAM intervention.

The family is therefore closed for unchanged deployable FAM routing. The closed
claim is limited to the tested FAM/FAM2 forms and selector evidence above; it is
not a claim that feature modulation can never work.

## Do Not Repeat Without New Evidence

- Do not rerun unchanged FAM/FAM2 modulation just because hard-bucket gains were
  large; the preservation failure is already documented.
- Do not treat average PSNR gains as sufficient when strong-reference
  regressions remain near `121/250` or worse.
- Do not launch another FAM selector unless its preflight shows stronger
  held-out separability than the failed selectivity-or-kill analysis.

## Reopen Condition

A FAM-family route can reopen only if a new deployable selector or preservation
guard passes a predeclared held-out diagnostic that directly measures false
intervention on strong-reference/easy images and retains meaningful hard-case
gain.
