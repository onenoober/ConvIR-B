# Hard-Frequency And Haze-Prior Family Summary

Date: 2026-06-04

Status: closed for the tested hard-frequency weighting and haze-prior SCM forms.

## Sources

- Index: `../EXPERIMENT_INDEX.md`
- Cards:
  - `../experiment_cards/2026-06-01-haze4k-hardfreq-loss.md`
  - `../experiment_cards/2026-06-01-haze4k-haze-prior-scm.md`
- Evidence roots:
  - `../experiment_logs/haze4k_hardfreq_loss_stop20_20260601/`
  - `../experiment_logs/haze4k_haze_prior_scm_20260601/`

## Established Facts

| Route | Main result | Decision |
| --- | --- | --- |
| Hard-aware frequency loss | Best mean PSNR `-0.2127 dB`; hard `+0.5999 dB`; easy `-1.2363 dB`; strong-reference regressions `188/250`; Best-vs-Last `-0.6922 dB`. | `FAIL_STOP_HARDFFT_LAMBDA_002`; do not repeat or promote `hard_fft_lambda=0.02` as-is. |
| Haze-prior SCM + hard auxiliary | Best mean PSNR `-0.3789 dB`; hard `+0.3501 dB`; easy `-1.6511 dB`; strong-reference regressions `185/250`. | `NO_PROMOTE_STOP20_HAZE_PRIOR_SCM_HARDAUX`; do not promote this exact route. |

## Family Verdict

Both tested routes moved hard cases in the intended direction but damaged global
mean, easy cases, and strong-reference preservation too much. The evidence
supports the idea that hard-focused signals can target weak samples, but the
current loss/auxiliary forms do not provide safe preservation.

This family is closed for the tested hard-frequency weight and haze-prior SCM
settings. The conclusion does not rule out future loss work that first proves
strong/easy protection before stop20.

## Do Not Repeat Without New Evidence

- Do not rerun `hard_fft_lambda=0.02` as-is.
- Do not repeat the exact haze-prior SCM + hard auxiliary configuration as a
  promotion route.
- Do not advance hard-positive loss variants when easy top-25% drops are near
  `-1 dB` or worse, or when strong-reference regressions remain above `180/250`.

## Reopen Condition

A future loss/prior route must predeclare explicit strong/easy protection and
show target-group gain plus preservation on a cheap diagnostic before any stop20
run. Mechanism metrics should include loss scale, gradient health, hard-bucket
gain, and strong/easy regression counts.
