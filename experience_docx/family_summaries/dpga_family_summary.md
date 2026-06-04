# DPGA Family Summary

Date: 2026-06-04

Status: active diagnostic family, not promotion-ready.

## Sources

- Index: `../EXPERIMENT_INDEX.md`
- Cards:
  - `../experiment_cards/2026-06-04-haze4k-convir-v1-0-dpga-lite.md`
  - `../experiment_cards/2026-06-04-haze4k-convir-v1-1-dpga-tail-control.md`
  - `../experiment_cards/2026-06-04-haze4k-convir-v1-3-hsdf.md`
- Evidence roots:
  - `../experiment_logs/haze4k_dpga_lite_20260604/`
  - `../experiment_logs/haze4k_dpga_tail_control_20260604/`
  - `../experiment_logs/haze4k_dpga_v13_hsdf_20260604/`

## Established Facts

| Route | Main result | Decision |
| --- | --- | --- |
| DPGA-Lite v1.0 | `Best.pkl` full-test mean `+0.0312 dB`, SSIM positive, hard `+0.0146 dB`, easy `+0.0209 dB`, strong-reference regressions `105/250`; exact `model_20`/`Final` mean `+0.0193 dB` and hard `+0.0037 dB`. | `DPGA_LITE_ADAPTER_ONLY_MIN_POSITIVE_BEST_BORDERLINE_FINAL`; directionally positive but not promotion-ready. |
| DPGA tail-control v1.1 | Shallow-only scale `0.25`, anchor `0.08`; Best mean `+0.037036 dB`, hard bottom-25% `+0.023367 dB`. | Failed `val_inner` hard gate `>= +0.030 dB`; locked Haze4K test blocked. |
| DPGA tail-control v1.2 | Shallow-only scale `0.5`, anchor `0.04`; Best mean `+0.042656 dB`, hard bottom-25% `+0.026225 dB`, worst `<= -0.20 dB` regressions rose to `16/300`. | Failed hard gate and worsened tail risk; no higher-scale follow-up without new diagnostic. |
| DPGA-v1.3A HSDF | Best `val_regular` mean `+0.026333 dB`; Best `val_hard` hard bottom-25 `+0.022099 dB`. | Loss-mask mechanism improved safety but missed hard gate; authorized only v1.3B diagnostic, not locked test. |
| DPGA-v1.3B HSDF | Best `val_regular` mean `+0.025839 dB`; Best `val_hard` hard bottom-25 `+0.023642 dB`; positive ratio `0.586667`; strong regression ratio `0.200000`; corrected bottleneck-only runtime ablation mean about `+0.000824 dB`. | `FAIL_STOP_V13B_HARD_GATED_BOTTLENECK`; locked test blocked. |

## Family Verdict

DPGA is the most active current family because it moved away from unsafe output
RGB residuals and places depth/prior information inside ConvIR feature paths.
DPGA-Lite v1.0 gave the first recent small positive full-test directional
signal without APDR output residuals, full-backbone training, FFT boost,
teacher distillation, or token-wise routing. That signal is still below the
current Haze4K noise-aware promotion standard and was partly test-observed, so
it is not a final improvement claim.

v1.1/v1.2 showed that shallow scale control can keep mean movement positive but
is hard-gain limited. v1.3 showed that hard-selective masking and hard-gated
bottleneck capacity did not deliver the needed hard-bottom gain, and corrected
runtime ablation found almost no useful bottleneck-only contribution.

The family remains open only for a new DPGA mechanism that directly addresses
hard-gain limitation without simply increasing scale or selecting on the locked
test.

## Do Not Repeat Without New Evidence

- Do not promote v1.0 from `Best.pkl` alone; exact stop20/final was borderline
  and the effect size is small relative to the route noise policy.
- Do not run locked Haze4K test for v1.1, v1.2, v1.3A, or v1.3B.
- Do not launch higher-scale shallow DPGA as the next step; v1.2 already raised
  worst-tail regressions to `16/300`.
- Do not continue the current HSDF hard-gated bottleneck route as-is; corrected
  ablation shows bottleneck-only mean contribution about `+0.000824 dB`.

## Reopen Condition

A DPGA follow-up must introduce a new hard-gain mechanism or diagnostic, not a
plain scale increase. It should select checkpoints/configuration on internal
validation or OOF-style protocols, then pass hard bottom-25%, regular/easy
safety, strong-reference regression, and worst-tail gates before any locked
Haze4K test.
