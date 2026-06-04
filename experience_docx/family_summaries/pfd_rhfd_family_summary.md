# PFD/RHFD Preservation Family Summary

Date: 2026-06-04

Status: diagnostic only; preservation improved in the rescue branch but hard-gain
and strong-reference gates failed.

## Sources

- Index: `../EXPERIMENT_INDEX.md`
- Cards:
  - `../experiment_cards/2026-06-02-haze4k-pfd-convir-mainline-plan.md`
  - `../experiment_cards/2026-06-02-haze4k-b1r-decoder-rhfd-preserve.md`
- Evidence roots:
  - `../experiment_logs/haze4k_pfd_mainline_20260602/`
  - `../experiment_logs/haze4k_b1r_decoder_rhfd_preserve_20260602/`

## Established Facts

| Route | Main result | Decision |
| --- | --- | --- |
| PFD mainline B1 stop20 | Stage 0 passed; B1 hard bottom-25% `+0.3838 dB`, global mean `-0.0885 dB`, easy top-25% `-0.3345 dB`, strong-reference regressions `137/250`. | Keep diagnostic; B1 fails preservation, so B2/B3 were not launched. |
| B1r decoder RHFD rescue | A0-level global delta `+0.0028 dB`, SSIM positive, easy top-25% `-0.0248 dB`, hard bottom-25% only `+0.0461 dB`, strong-reference regressions `103/250`. | `FAIL_STOP_B1R_DECODER_RHFD_ADAPTER_ONLY`; preservation improved over B1 but hard gain and strong-case gates fail. |

## Family Verdict

The PFD/RHFD sequence is useful as preservation evidence. B1 showed meaningful
hard-bucket movement but did not preserve global/easy/strong cases. B1r moved
closer to A0-level global behavior and reduced easy damage, but hard gain became
small and strong-reference regressions remained too high.

The current family is diagnostic only. It does not justify B2/B3 expansion or a
PFD promotion path without a new mechanism that recovers hard gain while keeping
B1r-like preservation.

## Do Not Repeat Without New Evidence

- Do not launch B2/B3 from the original PFD plan solely because B1 hard gain was
  positive; B1 failed the preservation gate.
- Do not treat B1r as candidate-positive: hard bottom-25% gain was only
  `+0.0461 dB` and strong-reference regressions were `103/250`.
- Do not add more PFD components unless the new component has a written
  preservation and hard-gain mechanism.

## Reopen Condition

A PFD/RHFD route can reopen if a new mechanism explains how it will retain the
B1r preservation improvement while materially increasing hard bottom-25% gain
and reducing strong-reference regressions on a predeclared validation protocol.
