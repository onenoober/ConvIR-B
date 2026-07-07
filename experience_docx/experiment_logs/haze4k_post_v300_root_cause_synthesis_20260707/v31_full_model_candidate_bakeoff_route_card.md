# Haze4K v3.1 Full-Model Candidate Bakeoff

Purpose: separate standard full-model quality from strict A0-dominance safe-upgrade after v3.0 closed the A0-anchored partial-unfreeze rescue route.

Route identity: diagnostic-only, train-derived full-image candidate bakeoff.

Fact sources:
- GitHub main v3.0 card and NoPost family summary for the closed A0-anchored route.
- Cloud `convir-4090` cached train-derived full-image per-image tables from v2.37 WDMamba, v2.39 ConvIR-L, and v2.2 FullUDP evidence.

Allowed:
- A0 official ConvIR-B baseline.
- Standalone WDMamba full-image Haze4K checkpoint.
- Standalone official ConvIR-L full-image Haze4K checkpoint.
- Standalone FullUDP table only if the joined 600-image contract matches A0.
- Compact JSON/CSV evidence.

Forbidden:
- no locked test;
- no canary80;
- no threshold or checkpoint chosen from locked test;
- no A0+alpha deployment contract;
- no selector, bridge, or generator;
- no v3.0 rescue by more decoder unfreezing, samples, folds, or loss tuning.

Metric contract:
- Split: 600 train-derived full-image Haze4K samples, joined by image name.
- Baseline: official ConvIR-B A0 same-context PSNR from the v2.37/v2.39 full-image cache.
- Hard/easy buckets: existing A0 same-context full-image buckets from v2.37/v2.39.
- Severe: delta vs A0 `<= -0.20 dB`.
- Strong-reference regression: strong-reference bucket and delta vs A0 `<= -0.05 dB`.

Result:
- WDMamba standalone: mean/hard/easy `+3.5778/+8.2765/-1.0483` dB, p05/CVaR5 `-2.9605/-4.0917` dB, severe `124`, strong-reference regressions `105`.
- ConvIR-L standalone: mean/hard/easy `+1.0945/+1.4339/+0.6015` dB, p05/CVaR5 `-1.9132/-3.0931` dB, severe `147`, strong-reference regressions `38`.
- FullUDP standalone: mean/hard/easy `-0.4313/+0.0049/-0.8842` dB; SSIM omitted because the historical source SSIM values fall outside `[0,1]`.
- Oracle over A0/WDMamba/ConvIR-L/FullUDP: mean delta `+4.4353 dB`, hard `+8.3350 dB`, easy `+1.2608 dB`, p05 `+0.0883 dB`, severe `0`.

Decision: `COMPLETED_GATE_PASS_WDMAMBA_FULL_MODEL_BASELINE_HEADROOM`.

Interpretation: v3.1 confirms full-model headroom is real and much larger than the A0-anchored residual/partial-unfreeze movement. It does not satisfy strict per-image A0 dominance because WDMamba and ConvIR-L still have tail regressions. The next authorized direction is v3.2 as a full model line, not A0 residual/selector/alpha rescue.

Evidence root: `experience_docx/experiment_logs/haze4k_v3_1_full_model_candidate_bakeoff_20260707/`.
