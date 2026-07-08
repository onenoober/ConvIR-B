# CHD-RM v2 Result Summary

Decision: `PAUSE_V2_DUAL_HEAD_NOT_PASSED`

Runtime source:

- Host: `convir-4090`
- Branch: `codex/haze4k-v5-v2-chd-rm-density-need-calibration`
- Run commit: `fffa0a4d08c8a7d1b2d07ad1da94c938da111973`
- Split: v1 fixed Haze4K train_inner 2400 / val_inner 600
- Locked Haze4K test: not used

Main results:

| Variant | Density Pearson | Density Spearman | Density AUROC | Need Pearson | Need Spearman | Need AUROC | Decision role |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| D0 handcrafted | 0.0249 | -0.0103 | 0.4848 | NA | NA | NA | failed lower-bound proxy |
| D1 dual head | 0.6715 | 0.6437 | 0.8925 | 0.1365 | 0.2198 | 0.6466 | density pass, need fail |
| D3 density-only | 0.6873 | 0.6628 | 0.9043 | NA | NA | NA | density signal confirmed |
| D4 need-only | NA | NA | NA | 0.1721 | 0.2508 | 0.6648 | near but below need gate |
| D5 shuffled control | -0.1336 | -0.1468 | 0.4118 | -0.2081 | -0.2578 | 0.3343 | control fails as required |

Interpretation:

- `H_density` is reliable enough for the next design layer.
- `R_need` is not reliable enough yet. It shows weak rank signal in D4, but the
  dual head does not pass Pearson/Spearman, and need strong-response coverage is
  zero under the current gate.
- v3 no-op RARM must remain blocked until `R_need` is repaired or replaced by a
  route-consistent calibration design.

Next allowed work:

- Audit whether the current `R_need_target = blur(gray_abs(O_A0 - I_gt))` is too
  sparse or too low-dynamic-range for direct sigmoid calibration.
- If keeping the same target, design a meaningful D2 that uses deeper features
  or a declared partial-unfreeze path without changing the dehazing output.
- Do not use the Haze4K locked test.
