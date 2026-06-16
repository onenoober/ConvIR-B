# Haze4K v2.6 Residual Shrinkage Alpha Curves Decision

Decision: `V26_ALPHA_CURVES_COMPLETED_LOCKED_UNTOUCHED`

This supplemental route evaluates fixed anchor-preserving residual shrinkage on the C8 train-derived `val_regular + val_hard` scope only. It does not read or write locked Haze4K evidence and does not tune from the prior WD0375 locked result.

## Main Readout

- WDMamba: safe positive/tail alpha set `0.125, 0.25, 0.375, 0.5`; endpoint severe `124.0/600`; best safety alpha `0.125`.
- FSNet+UDP: safe positive/tail alpha set `0.125, 0.25, 0.375, 0.5, 0.75`; endpoint severe `71.0/600`; best safety alpha `0.125`.
- MB-TaylorFormerV2-L: safe positive/tail alpha set `0.125`; endpoint severe `294.0/600`; best safety alpha `0.125`.

## Evidence Files

- `v26_all_expert_alpha_grid.csv`
- `v26_all_expert_group_min.csv`
- `v26_compact_comparison.csv`
- `v26_summary.json`

Locked-test status: `locked_test_touched=false`.
