# PFD/RHFD Preservation Family Summary

Date: 2026-06-07

Status: diagnostic only; B1-Surgery and SafeRHFD variants improved preservation
relative to B1, but hard-gain robustness and strong-reference gates still fail.

## Sources

- Index: `../EXPERIMENT_INDEX.md`
- Cards:
  - `../experiment_cards/2026-06-02-haze4k-pfd-convir-mainline-plan.md`
  - `../experiment_cards/2026-06-02-haze4k-b1r-decoder-rhfd-preserve.md`
  - `../experiment_cards/2026-06-02-haze4k-b1-surgery-preserve.md`
  - `../experiment_cards/2026-06-02-haze4k-saferhfd-v2-stage-scale.md`
  - `../experiment_cards/2026-06-02-haze4k-saferhfd-v2-train.md`
- Evidence roots:
  - `../experiment_logs/haze4k_pfd_mainline_20260602/`
  - `../experiment_logs/haze4k_b1r_decoder_rhfd_preserve_20260602/`
  - `../experiment_logs/haze4k_b1_surgery_preserve_20260602/`
  - `../experiment_logs/haze4k_saferhfd_v2_stage_scale_20260602/`
  - `../experiment_logs/haze4k_saferhfd_v2_train_20260602/`

## Established Facts

| Route | Main result | Decision |
| --- | --- | --- |
| PFD mainline B1 stop20 | Stage 0 passed; B1 hard bottom-25% `+0.3838 dB`, global mean `-0.0885 dB`, easy top-25% `-0.3345 dB`, strong-reference regressions `137/250`. | Keep diagnostic; B1 fails preservation, so B2/B3 were not launched. |
| B1r decoder RHFD rescue | A0-level global delta `+0.0028 dB`, SSIM positive, easy top-25% `-0.0248 dB`, hard bottom-25% only `+0.0461 dB`, strong-reference regressions `103/250`. | `FAIL_STOP_B1R_DECODER_RHFD_ADAPTER_ONLY`; preservation improved over B1 but hard gain and strong-case gates fail. |
| B1-Surgery preserve sweep | A0 backbone plus scaled B1 RHFD branches. Scale `0.70` had mean `+0.01064 dB`, hard `+0.03317 dB`, easy `+0.00782 dB`, severe `0`, strong regressions `0`; scale `1.00` had slightly higher gain but more regressions. | `B1_SURGERY_DIAGNOSTIC_POSITIVE_NOT_PROMOTION_READY`; useful preservation-safe signal, but too thin and test-derived. |
| SafeRHFD-v2 stage-scale | 11 no-training RHFD1/RHFD2 scale candidates; strict passing candidates `0`. Best failed candidate `RHFD2=0.50,RHFD1=0.70` had mean `+0.00779 dB`, hard `+0.02228 dB`, easy `+0.00693 dB`, but severe `1`, hard median `-0.00136`, hard positive ratio `0.44`. | `FAIL_STRICT_ROBUSTNESS_GATE`; do not promote or launch B2/B3 from this evidence. |
| SafeRHFD-v2 train | Independent stop20 training had mean `+0.00568 dB`, SSIM `+0.000024`, hard `+0.04890 dB`, easy `-0.00920 dB`, strong regressions `70/250`, severe `18/1000`. | `FAIL_STOP_SAFERHFD_V2_TRAIN`; missed mean/hard gates and strong-reference limit. |

## Family Verdict

The PFD/RHFD sequence is useful as preservation evidence. B1 showed meaningful
hard-bucket movement but did not preserve global/easy/strong cases. B1r moved
closer to A0-level global behavior and reduced easy damage, but hard gain became
small and strong-reference regressions remained too high. B1-Surgery showed
that B1 RHFD branches can be reattached to A0 with small positive movement, but
the stricter SafeRHFD-v2 stage-scale and train diagnostics did not clear the
robustness or strong-reference gates.

The current family is diagnostic only. It does not justify B2/B3 expansion or a
PFD promotion path without a new mechanism that recovers hard gain while keeping
B1r-like preservation.

## Do Not Repeat Without New Evidence

- Do not launch B2/B3 from the original PFD plan solely because B1 hard gain was
  positive; B1 failed the preservation gate.
- Do not treat B1r as candidate-positive: hard bottom-25% gain was only
  `+0.0461 dB` and strong-reference regressions were `103/250`.
- Do not promote B1-Surgery `0.70` or SafeRHFD-v2 from single-route positive
  movement; the stricter robustness and train gates failed.
- Do not add more PFD components unless the new component has a written
  preservation and hard-gain mechanism.

## Reopen Condition

A PFD/RHFD route can reopen if a new mechanism explains how it will retain the
B1r preservation improvement while materially increasing hard bottom-25% gain
and reducing strong-reference regressions on a predeclared validation protocol.
