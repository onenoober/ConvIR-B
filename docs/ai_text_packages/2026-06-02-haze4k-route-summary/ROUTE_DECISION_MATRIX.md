# Haze4K Route Decision Matrix

Date: 2026-06-02

Status: compact summary of completed Haze4K route evidence through
SafeRHFD-v2.

## Current Verdict

No trained Haze4K replacement route is promotion-ready.

The strongest repeated pattern is that hard-sample quality can be moved, but
the tested interventions damage easy or strong-reference cases too often.
No-training B1 surgery found a safer preservation-first checkpoint surgery
candidate at `scale=0.70`; SafeRHFD-v2 training was safer than full-model RHFD
but still failed the automatic replacement gate.

## Route Matrix

| Route | Best global signal | Hard bucket signal | Main failure or limit | Decision |
| --- | ---: | ---: | --- | --- |
| FAM `modres` stop5 | `+0.0953 dB` mean PSNR | not bucketed in the card | median delta negative; strong-reference regressions `142/250` | Diagnostic only; do not promote unchanged `modres`. |
| FAM2-only stop20 | `+0.1739 dB` mean PSNR | `+0.8159 dB` hard bottom 25% | easy top 25% `-0.2860 dB`; strong-reference regressions `138/250` | Diagnostic only; hard signal real, preservation fails. |
| FAM2 bounded gamma | `-0.0271 dB` mean PSNR | `+0.8054 dB` hard bottom 25% | easy top 25% `-1.2740 dB`; strong-reference regressions `181/250` | Stop bounded gamma as a promotion route. |
| FAM2 confidence-gated gamma | `+0.4523 dB` mean PSNR | `+0.9380 dB` hard bottom 25% | strong-reference regressions `121/250`; deployable gate not selective enough | Positive diagnostic signal; not decision-grade. |
| Stop20 original noise floor | seed mean PSNR std `0.2206 dB` | hard bucket std `0.4551 dB` | single-seed deltas are noisy | Use as route-decision noise floor. |
| FAM2 selectivity-or-kill | best feasible threshold-gate mean `+0.1333 dB` | selector analysis only | no deployable selector passes AUC and threshold gates | `FAIL_STOP_FAM_ROUTE`. |
| Hard-aware frequency loss | `-0.2127 dB` mean PSNR | `+0.5999 dB` hard bottom 25% | easy top 25% `-1.2363 dB`; stability `-0.6922 dB` Best-vs-Last | `FAIL_STOP_HARDFFT_LAMBDA_002`. |
| Haze-prior SCM + hard auxiliary | `-0.3789 dB` mean PSNR | `+0.3501 dB` hard bottom 25% | easy top 25% `-1.6511 dB`; strong-reference regressions `185/250` | `NO_PROMOTE_STOP20_HAZE_PRIOR_SCM_HARDAUX`. |
| PFD mainline B1 stop20 | `-0.0885 dB` mean delta | `+0.3838 dB` hard bottom 25% | easy top 25% `-0.3345 dB`; strong-reference regressions `137/250` | Keep as diagnostic; do not launch B2/B3 from this B1 as-is. |
| B1 surgery `scale=0.70` | `+0.01064 dB` mean delta | `+0.03317 dB` hard bottom 25% | no-training checkpoint surgery, not a trained route; small median gain | Preservation-first diagnostic candidate; use as surgery evidence, not promotion. |
| B1 surgery `scale=1.00` | `+0.01268 dB` mean delta | `+0.03888 dB` hard bottom 25% | more global/easy/strong-reference regressions than `scale=0.70` | High-gain backup evidence only. |
| SafeRHFD-v2 pfd-only stop20 | `+0.00568 dB` mean delta | `+0.04890 dB` hard bottom 25% | global and hard gains below gate; strong-reference regressions `70/250` | Keep as diagnostic; automatic gate failed. |

## Practical Conclusion

- Keep the central index as the reading entry point.
- Keep retained leaf branches only for runnable experiment snapshots.
- Do not merge failed diagnostic route code into stable `main`.
- Do not repeat FAM2 selector, `hard_fft_lambda=0.02`, exact haze-prior SCM
  hard-aux, PFD B1 full-model RHFD, or SafeRHFD-v2 pfd-only stop20 as-is.
- Any future route must predefine easy-case preservation, strong-reference
  regression, stability, and deployable-signal gates before cloud training.

## Reopen Conditions

A route can be reopened only if it introduces a new deployable signal, surgery
rule, or training objective that passes a cheap preflight and explicitly
reduces collateral damage on easy and strong-reference images. Hard-bucket
improvement alone is not enough.
