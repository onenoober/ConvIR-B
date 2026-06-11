# DTA Family Summary

Date: 2026-06-11

Status: first low-gate route completed diagnostic/no-promotion on
`convir-4090`; DTA-v2 calibrated confidence-gated route is active with
adapter-only five-fold OOF evidence synced; multi-seed controls for seeds
`3411` and `3413` are running, and locked test remains blocked.

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
| DTA-v2 calibrated confidence-gated adapter-only | Fold0 OOF20 controls completed: invert/normal/shuffle/zero mean dPSNR `+0.1069/+0.1060/+0.0984/+0.0955`, hard `+0.0992/+0.1047/+0.0956/+0.0918`. Positive but weak depth attribution. | Keep as current internal candidate; expand OOF folds before locked-test consideration. |
| DTA-v2 adapter-only five-fold OOF | Five-fold OOF20 completed over `3000` train-derived validation predictions. Invert/normal/shuffle/zero mean dPSNR `+0.0882/+0.0879/+0.0765/+0.0728`; hard `+0.0700/+0.0763/+0.0633/+0.0595`; easy `+0.0716/+0.0697/+0.0649/+0.0630`; bootstrap lower bounds are all positive. | Positive internal signal, but not promotion-ready: zero/shuffle retain most gain, raw `normal` nearly ties calibrated `invert`, SSIM is slightly negative, and worst regressions remain high. Continue multi-seed controls. |
| DTA-v2 adapter-only multi-seed controls | Seeds `3411` and `3413` launched as five-fold OOF20 adapter-only controls on convir-4090 GPUs `1-7`; GPU0 is intentionally skipped due non-DTA user processes. | Running; aggregate with seed `3407` after completion before any locked-test decision. |
| DTA-v2 calibrated confidence-gated adapter-neighbors | Fold0 OOF20 controls completed: invert/normal/shuffle/zero mean dPSNR `+0.0151/+0.0151/+0.0097/+0.0072`, easy top-25 `-0.0639/-0.0624/-0.0728/-0.0746`, negative SSIM, and worst regressions `142-146`. | Not a promotion candidate; do not continue adapter-neighbors unless a new mechanism changes preservation/gate behavior. |

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
   Fold0 adapter-neighbors is now complete and negative for preservation, so the
   active path returns to adapter-only OOF expansion.
6. Use locked Haze4K test only once for a fixed internally selected config.
