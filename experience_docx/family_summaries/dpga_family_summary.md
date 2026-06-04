# DPGA Family Summary

Date: 2026-06-04

Status: active diagnostic family, not promotion-ready.

## Sources

- Index: `../EXPERIMENT_INDEX.md`
- Cards:
  - `../experiment_cards/2026-06-04-haze4k-convir-v1-0-dpga-lite.md`
  - `../experiment_cards/2026-06-04-haze4k-convir-v1-1-dpga-tail-control.md`
  - `../experiment_cards/2026-06-04-haze4k-convir-v1-3-hsdf.md`
  - `../experiment_cards/2026-06-04-haze4k-convir-v1-4-udp-lite.md`
  - `../experiment_cards/2026-06-04-haze4k-convir-v1-4b-bidpfm1.md`
- Evidence roots:
  - `../experiment_logs/haze4k_dpga_lite_20260604/`
  - `../experiment_logs/haze4k_dpga_tail_control_20260604/`
  - `../experiment_logs/haze4k_dpga_v13_hsdf_20260604/`
  - `../experiment_logs/haze4k_udp_lite_v14_20260604/`
  - `../experiment_logs/haze4k_udp_lite_v14b_bidpfm1_20260604/`

## Established Facts

| Route | Main result | Decision |
| --- | --- | --- |
| DPGA-Lite v1.0 | `Best.pkl` full-test mean `+0.0312 dB`, SSIM positive, hard `+0.0146 dB`, easy `+0.0209 dB`, strong-reference regressions `105/250`; exact `model_20`/`Final` mean `+0.0193 dB` and hard `+0.0037 dB`. | `DPGA_LITE_ADAPTER_ONLY_MIN_POSITIVE_BEST_BORDERLINE_FINAL`; directionally positive but not promotion-ready. |
| DPGA tail-control v1.1 | Shallow-only scale `0.25`, anchor `0.08`; Best mean `+0.037036 dB`, hard bottom-25% `+0.023367 dB`. | Failed `val_inner` hard gate `>= +0.030 dB`; locked Haze4K test blocked. |
| DPGA tail-control v1.2 | Shallow-only scale `0.5`, anchor `0.04`; Best mean `+0.042656 dB`, hard bottom-25% `+0.026225 dB`, worst `<= -0.20 dB` regressions rose to `16/300`. | Failed hard gate and worsened tail risk; no higher-scale follow-up without new diagnostic. |
| DPGA-v1.3A HSDF | Best `val_regular` mean `+0.026333 dB`; Best `val_hard` hard bottom-25 `+0.022099 dB`. | Loss-mask mechanism improved safety but missed hard gate; authorized only v1.3B diagnostic, not locked test. |
| DPGA-v1.3B HSDF | Best `val_regular` mean `+0.025839 dB`; Best `val_hard` hard bottom-25 `+0.023642 dB`; positive ratio `0.586667`; strong regression ratio `0.200000`; corrected bottleneck-only runtime ablation mean about `+0.000824 dB`. | `FAIL_STOP_V13B_HARD_GATED_BOTTLENECK`; locked test blocked. |
| ConvIR-Dehaze-v1.4-UDP-Lite | v1.4A adapter-only completed and failed gate: Best `val_regular` mean `+0.028294 dB`, Best `val_hard` mean `+0.020340 dB`, hard bottom-25 `+0.022275 dB`, positive ratio `0.586667`, worst count `19`. Ablation shows `DPFM1-only` is safer/stronger (`val_hard` mean `+0.026774 dB`, worst `0`) while `DPFM2-only` is negative. | `FAIL_V14A_ADAPTER_ONLY_FULL_DPFM123`; locked test blocked. Do not micro-tune full DPFM123 scale/gate; only DPFM1-focused diagnostic or v1.4B fusion-neighbor partial unfreeze is evidence-supported. |
| ConvIR-Dehaze-v1.4B-BiDPFM1 | `udp_bi`, `active_adapters=dpfm1`, `active_adapter_only` completed. Best `val_regular` mean `+0.028624 dB`, positive ratio `0.536667`, worst count `17`, strong ratio `0.28`; Best `val_hard` mean `+0.023429 dB`, hard bottom-25 `+0.020760 dB`, worst count `8`. | `FAIL_STOP_V14B_BIDPFM1_ADAPTER_ONLY`; locked test blocked; do not rerun BiDPFM1-only scale/gate tuning. |

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

v1.4-UDP-Lite tested the currently preferred reopen mechanism: zero-init
multi-scale depth/prior fusion (`DPGA_prior_encoder`, `DPGA_dpfm1/2/4`) with
independent zero-init, module-ablation, and depth-quality audit tooling. The
cloud A0-equivalence preflight passed, but v1.4A adapter-only failed the
internal regular+hard gate. The most useful evidence is scale attribution:
`DPFM1-only` is the only strong/safe contributor, full `DPFM1+2+4` raises tail
risk, and `DPFM2-only` is a negative contributor.

v1.4B-BiDPFM1 was the authorized DPFM1-focused follow-up. Its `udp_bi`
A0-equivalence and projection-gradient liveness preflight passed, but
adapter-only training did not clear the internal continue line. The no-training
matrix found `DPFM1+4` has better mean than DPFM1-only but not a clean enough
tail profile for the first route, while DPFM2 remains blocked. The completed
BiDPFM1-only route is stopped; this is not permission to run locked Haze4K test,
revive DPFM2, or perform full multi-scale scale search.

## Do Not Repeat Without New Evidence

- Do not promote v1.0 from `Best.pkl` alone; exact stop20/final was borderline
  and the effect size is small relative to the route noise policy.
- Do not run locked Haze4K test for v1.1, v1.2, v1.3A, or v1.3B.
- Do not launch higher-scale shallow DPGA as the next step; v1.2 already raised
  worst-tail regressions to `16/300`.
- Do not continue the current HSDF hard-gated bottleneck route as-is; corrected
  ablation shows bottleneck-only mean contribution about `+0.000824 dB`.
- Do not treat v1.4 as permission to run locked Haze4K test; v1.4A failed
  internal `val_regular`/`val_hard` gates.
- Do not micro-tune full `DPFM1+2+4` scale/gate after v1.4A; ablation shows
  `DPFM2-only` is negative and full DPFM123 increases tail risk.
- Do not run locked Haze4K test for v1.4B before the written internal
  regular+hard gate passes.

## Reopen Condition

A DPGA follow-up must introduce a new hard-gain mechanism or diagnostic, not a
plain scale increase. It should select checkpoints/configuration on internal
validation or OOF-style protocols, then pass hard bottom-25%, regular/easy
safety, strong-reference regression, and worst-tail gates before any locked
Haze4K test.
