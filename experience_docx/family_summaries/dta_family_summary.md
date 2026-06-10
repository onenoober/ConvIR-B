# DTA Family Summary

Date: 2026-06-11

Status: first depth-guided transmission adapter route completed on
`convir-4090`; diagnostic gate passed under lenient rules, but the exact
low-gate adapter is not promotion-ready.

## Sources

- Index: `../EXPERIMENT_INDEX.md`
- Card: `../experiment_cards/2026-06-10-haze4k-dta-lowgate.md`
- Evidence root: `../experiment_logs/haze4k_dta_lowgate_20260610/`
- Runnable branch: `github/codex/haze4k-dta-lowgate`

## Established Facts

| Route | Main result | Decision |
| --- | --- | --- |
| DTA low-gate preflight | A0 partial load passed with `602` keys loaded, `18` missing keys all under `DTA.`, unexpected keys `[]`; synthetic no-op max diff `0.0`; real-batch DTA grad sum `0.02175214`. | Implementation and initialization contract are valid. |
| DTA smoke | One-epoch adapter-only smoke completed; 32-image diagnostic mean PSNR delta `+0.002904 dB`, strong regressions `0`, worst regressions `0`. | Continue to scout because runtime, depth cache, and low-gate path are healthy. |
| DTA scout5 | Five-epoch adapter-only run completed; 128-image diagnostic mean PSNR delta `-0.036217 dB`, hard bottom-25 `-0.039902 dB`, strong regressions `15/32`, worst regressions `0`. | Passed the lenient continuation gate, but did not show a positive mechanism result. |
| DTA gate20 | Twenty-epoch adapter-only run completed; full 1000-image diagnostic mean PSNR delta `-0.008940 dB`, hard bottom-25 `-0.019101 dB`, easy top-25 `-0.021037 dB`, SSIM delta `-0.00001973`, strong regressions `80/250`, worst regressions `48/1000`. | `COMPLETED_GATE_PASS_DIAGNOSTIC_NO_PROMOTION_DTA_LOWGATE`; do not promote this exact route. |

## Family Verdict

DTA validates the engineering path for Innovation 1: cached relative depth can
be loaded with synchronized augmentation, the official ConvIR-B checkpoint can
be partially loaded into a DTA branch, zero/no-op initialization is exact, and
adapter-only training runs normally on `convir-4090`.

The scientific result is neutral-to-negative. The low gates behaved as
requested and remained bounded, and the transmission rank loss decreased
(`0.6472` to `0.5304` by epoch 20), so the auxiliary mechanism is live. However,
the full diagnostic comparison stayed slightly below A0 and did not produce the
target hard/far-scene gain. Tail risk also remains visible through
`80/250` strong-reference regressions and `48/1000` worst regressions at the
`<= -0.20 dB` threshold.

Do not repeat this exact adapter-only low-gate DTA route as a promotion attempt.
Future DTA work needs a new card and should change the mechanism rather than
only retuning the same gate/loss schedule on the full diagnostic result.
