# CHD-RM v2d Overall Result Summary

Decision: `PAUSE_V2D_D7C_TOPK_PROMISING_BUT_CONTROLS_WEAK_NO_V3`

Runtime source:

- Host: `convir-4090`
- Branch: `codex/haze4k-v5-v2d-chd-rm-need-spatial-hard-negative`
- Base commit: `1f088dc2d0a15659afce86820dfe8fa2ee7c0aa3`
- Split: fixed Haze4K train_inner 2400 / val_inner 600
- Locked Haze4K test usage: none
- D2/RARM/v3: not run

## Key Result

D7c frozen multi-context need head is the best current direction. D7c top-k passes the candidate ranking and spatial safety gate under a train_inner-derived threshold, but controls are not clean enough to enter v3 no-op RARM yet.

| Stage | Variant | Spearman | AUROC | AUPRC | Coverage | Recall | False global | False p90 | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| D7-0 | frozen D6c raw threshold | 0.3298 | 0.7133 | 0.4353 | 0.2530 | 0.3402 | 0.0988 | 0.3288 | global-safe only, per-image unsafe |
| D7a | HN ordinal trained head | 0.3024 | 0.6983 | 0.4142 | 0.2681 | 0.3325 | 0.0988 | 0.2680 | not enough |
| D7b | top-k HN trained head | 0.2950 | 0.6925 | 0.4178 | 0.2529 | 0.3182 | 0.0992 | 0.2537 | not enough |
| D7c | multi-context HN | 0.5320 | 0.8527 | 0.6524 | 0.3027 | 0.4602 | 0.0139 | 0.0865 | candidate pass, p95 caveat |
| D7c | multi-context top-k HN | 0.5175 | 0.8456 | 0.6442 | 0.3027 | 0.4493 | 0.0030 | 0.0246 | best candidate |

## Controls

| Control | Spearman | AUROC | AUPRC | Coverage | False p90 | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| D7c shuffled target | 0.2397 | 0.6593 | 0.5029 | 0.3011 | 0.0000 | fails full R_need ranking gate, but not a near-zero shuffle failure |
| D7c random target | 0.2356 | 0.6529 | 0.5116 | 0.3023 | 0.0000 | same weak density/context proxy behavior |
| D3 density-only control | 0.2958 | 0.6890 | 0.4336 | 0.1045 | 0.0051 | fails full gate via coverage/recall/ranking |

## Interpretation

D7a/D7b confirm the v2c failure is not fixed by simply adding hard-negative loss to the shallow res1 head. D7c shows the missing ingredient is frozen multi-context localization, not D2 or RARM.

However, shuffled/random controls retain weak positive ranking, likely because density/context proxies remain correlated with residual need. Therefore D7c top-k is a strong candidate, but not yet permission to enter v3 no-op RARM.

## Next Allowed Work

1. Keep D7c top-k as the current candidate route.
2. Run stricter controls before any v3/no-op RARM audit:
   - fixed image-level permutation with target/mask pairing recorded;
   - density-only matched-threshold control;
   - random-target control with expected near-zero rank criterion.
3. Add a recall-protection audit for low-density high-need regions. D7c top-k has excellent false-strong safety, but selected low-density high-need recall is only `0.0370`; D7c HN is higher at `0.1160` with weaker tail safety.
4. Do not run D2, connect/train RARM, use v3 expansion, or touch the locked Haze4K test.
