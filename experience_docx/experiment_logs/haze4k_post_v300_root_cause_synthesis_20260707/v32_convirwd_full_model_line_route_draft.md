# v3.2 ConvIR-WD Full Model Line Draft

Purpose:
Build or fine-tune an end-to-end non-post dehazing model that uses full-model capacity plus wavelet/low-frequency haze modeling, without A0 output anchoring.

Starting rule:
Any ConvIR architecture change must branch from immutable `github/codex/haze4k-official-arch-anchor` and follow `Haze4K_ARCH_FINETUNE_WORKFLOW.md`.

Architecture direction:
- ConvIR-B/L-scale backbone or WDMamba-informed low-frequency branch.
- Wavelet low-frequency restoration branch.
- High-frequency detail refinement branch.
- Global context block in low-resolution/LL path.
- Optional YCbCr/Y-channel structural guidance.
- Optional haze-state auxiliary heads.
- No final-output postprocess.
- No A0+teacher alpha.
- No deployable selector as the first-class route.

Protocol:
- P0 architecture/preflight: official baseline untouched; locked test untouched; deterministic eval scripts; trainable manifest; parameter/FLOP budget.
- P1 mini-overfit sanity on 8 or 16 train-derived images; no quality claim.
- P2 train-derived validation with a real split such as 480/120 or large OOF folds; checkpoint selected only on this split.
- P3 fixed candidate confirmation only after P2 passes.
- P4 locked test one-shot only after fixed candidate selection.

Primary gate:
Use model-line success, not strict A0 per-image dominance.

Model-line success gate:
- mean delta vs A0 `>= +0.30 dB`;
- hard delta vs A0 `>= +0.50 dB`;
- easy delta vs A0 `>= -0.05 dB`;
- p05 delta vs A0 `>= -0.30 dB`;
- CVaR5 delta vs A0 `>= -0.50 dB`;
- no catastrophic visual failures;
- beats A0 and is Pareto-competitive with WDMamba/ConvIR-L candidate baselines.

Locked test remains blocked until P2/P3 select a fixed candidate.
