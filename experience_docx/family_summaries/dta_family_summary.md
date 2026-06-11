# DTA Family Summary

Date: 2026-06-11

Status: first low-gate route completed diagnostic/no-promotion on
`convir-4090`; DTA-v2 calibrated confidence-gated route is active and pending
cloud audit/preflight/training evidence.

## Sources

- Index: `../EXPERIMENT_INDEX.md`
- Card: `../experiment_cards/2026-06-10-haze4k-dta-lowgate.md`
- Evidence root: `../experiment_logs/haze4k_dta_lowgate_20260610/`
- Runnable branch: `github/codex/haze4k-dta-lowgate`
- Active reopened branch: `github/codex/haze4k-dta-v2-calibrated`

## Established Facts

| Route | Main result | Decision |
| --- | --- | --- |
| DTA low-gate preflight | A0 partial load passed with `602` keys loaded, `18` missing keys all under `DTA.`, unexpected keys `[]`; synthetic no-op max diff `0.0`; real-batch DTA grad sum `0.02175214`. | Implementation and initialization contract are valid. |
| DTA smoke | One-epoch adapter-only smoke completed; 32-image diagnostic mean PSNR delta `+0.002904 dB`, strong regressions `0`, worst regressions `0`. | Continue to scout because runtime, depth cache, and low-gate path are healthy. |
| DTA scout5 | Five-epoch adapter-only run completed; 128-image diagnostic mean PSNR delta `-0.036217 dB`, hard bottom-25 `-0.039902 dB`, strong regressions `15/32`, worst regressions `0`. | Passed the lenient continuation gate, but did not show a positive mechanism result. |
| DTA gate20 | Twenty-epoch adapter-only run completed; full 1000-image diagnostic mean PSNR delta `-0.008940 dB`, hard bottom-25 `-0.019101 dB`, easy top-25 `-0.021037 dB`, SSIM delta `-0.00001973`, strong regressions `80/250`, worst regressions `48/1000`. | `COMPLETED_GATE_PASS_DIAGNOSTIC_NO_PROMOTION_DTA_LOWGATE`; do not promote this exact route. |
| DTA-v2 calibrated confidence-gated | Pending cloud audit/preflight/training. Implements depth-transmission audit, calibrated six-channel prior, confidence-gated bounded FiLM, zero-init decoder residual, supervised transmission/physics/preservation losses, and true/zero/shuffle/invert controls. | `IN_PROGRESS_CLOUD_QUEUE_PENDING`; execute on `convir-4090` and keep locked Haze4K test blocked until one fixed internal-selected configuration. |

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


## Active DTA-v2 Queue

DTA-v2 is the authorized reopen route because it changes the mechanism rather
than retuning the completed low-gate adapter. Required sequence:

1. Audit cached depth against Haze4K `trans` maps for direction, alpha, proxy
   error, and low-texture/bright/dense-region risk.
2. Run DTA-v2 preflight with A0 partial load, no-op equivalence, real-batch
   gradients, supervised transmission, and physics loss probes.
3. Train adapter-only on train-derived internal/OOF splits with true depth.
4. Run zero-depth, shuffle-depth, and invert-depth controls under the same
   protocol.
5. Run adapter-neighbors only after the adapter-only/control evidence is synced.
6. Use locked Haze4K test only once for a fixed internally selected config.
