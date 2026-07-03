# Hard-Frequency And Haze-Prior Family Summary

Date: 2026-07-03

Status: closed for the tested hard-frequency weighting, haze-prior SCM, and
current NoPost O2 spatial lowband learnability forms.

## Sources

- Index: `../EXPERIMENT_INDEX.md`
- Cards:
  - `../experiment_cards/2026-06-01-haze4k-hardfreq-loss.md`
  - `../experiment_cards/2026-06-01-haze4k-haze-prior-scm.md`
- Evidence roots:
  - `../experiment_logs/haze4k_hardfreq_loss_stop20_20260601/`
  - `../experiment_logs/haze4k_haze_prior_scm_20260601/`
  - `../experiment_logs/haze4k_v2_19_nopost_spatial_lowband_policy_learnability_20260703/`

## Established Facts

| Route | Main result | Decision |
| --- | --- | --- |
| Hard-aware frequency loss | Best mean PSNR `-0.2127 dB`; hard `+0.5999 dB`; easy `-1.2363 dB`; strong-reference regressions `188/250`; Best-vs-Last `-0.6922 dB`. | `FAIL_STOP_HARDFFT_LAMBDA_002`; do not repeat or promote `hard_fft_lambda=0.02` as-is. |
| Haze-prior SCM + hard auxiliary | Best mean PSNR `-0.3789 dB`; hard `+0.3501 dB`; easy `-1.6511 dB`; strong-reference regressions `185/250`. | `NO_PROMOTE_STOP20_HAZE_PRIOR_SCM_HARDAUX`; do not promote this exact route. |
| NoPost spatial lowband v2.19 | P0 source-clean/identity passed. P1 small-CNN spatial predictor had mean `+0.9921 dB`, hard `+2.6504 dB`, and positive ratio `0.7192`, but easy `-0.1346 dB`, p05 `-1.1486 dB`, CVaR5 `-2.1058 dB`, severe rate `0.2025`, strong-reference regressions `302/600`, and fold tail pass count `0/5`. P3 guard replay passed. | `V219_LEARNABILITY_FAIL_OR_GUARD_FAIL_PAUSE_BEFORE_TRAINING`; do not train WLDB-B from current O2 spatial final-feature LL predictor form. |

## Family Verdict

Both tested routes moved hard cases in the intended direction but damaged global
mean, easy cases, and strong-reference preservation too much. The evidence
supports the idea that hard-focused signals can target weak samples, but the
current loss/auxiliary forms do not provide safe preservation.

This family is closed for the tested hard-frequency weight, haze-prior SCM
settings, and current O2 spatial lowband predictor form. The conclusion does
not rule out future loss or lowband-context work that first proves strong/easy
protection before stop20 or training.

## Do Not Repeat Without New Evidence

- Do not rerun `hard_fft_lambda=0.02` as-is.
- Do not repeat the exact haze-prior SCM + hard auxiliary configuration as a
  promotion route.
- Do not train WLDB-B from the current O2 spatial final-feature LL predictor
  form after v2.19; P1 failed tail/easy/strong preservation despite mean and
  hard gains.
- Do not advance hard-positive loss variants when easy top-25% drops are near
  `-1 dB` or worse, or when strong-reference regressions remain above `180/250`.

## Reopen Condition

A future loss/prior/lowband route must predeclare explicit strong/easy
protection and show target-group gain plus preservation on a cheap diagnostic
before any stop20 run or training launch. For lowband policy routes, the next
material reopen condition is a context-rich learnability audit, such as O3
mid+final/context signal, that fixes the v2.19 tail/easy failure.
