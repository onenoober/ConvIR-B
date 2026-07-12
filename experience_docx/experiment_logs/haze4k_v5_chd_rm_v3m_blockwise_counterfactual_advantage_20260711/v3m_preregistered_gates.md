# v3m-A0a Preregistered Gates

Primary baseline: fixed `alpha=0.125` for the same frozen operator and the
same train-derived clean-reference grouped OOF rows.

Common action set: `{0, 0.125, 0.25, 0.5, 1.0}`.

Primary policy comparison: `ORACLE_BLOCK16_GRID` against
`ORACLE_PIXEL_GRID`, using lift beyond fixed `alpha=0.125`.

For each of `D_ref` and `D_rep`, pass requires:

1. grouped paired mean-lift CI95 low `> 0`;
2. block16 retention CI95 low `>= 0.80`;
3. block16 p10 `>=` reference p10;
4. block16 worst `>=` reference worst;
5. block16 severe count `<=` reference severe count.

No diagnostic result can alter this action set, denominator, threshold, or
comparison scope after the run starts.
