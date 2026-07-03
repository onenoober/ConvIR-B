# v2.17 R2 Capacity Ladder Decision

Decision: `R2_O1_GLOBAL_FEATURE_LL_PASS_REVIEW_WLDB_A2_OBJECTIVE`

- O1 pass: `True`
- O2 pass: `True`
- O3 pass: `True`
- O4 pass: `True`
- Best mean variant: `O4_rgb_ll_reference`

Interpretation:

- If only O2/O3 passes, close WLDB-A but keep spatial/internal feature-lowband open.
- If no internal feature oracle passes while O4 passes, RGB LL headroom did not transfer to frozen internal feature correction.
- Locked Haze4K test remains untouched.
