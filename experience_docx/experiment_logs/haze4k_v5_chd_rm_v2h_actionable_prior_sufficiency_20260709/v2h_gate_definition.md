# v2h Gate Definition

## v2h-A Risk-Coverage Gate

Primary D7c operating point should satisfy on val-inner:

- selected coverage in `[0.25, 0.35]`;
- action recall at least the v2g D7c fixed point minus `0.01`, or the fixed D7c point itself remains valid;
- low-adjacent recall `>= 0.15`;
- global negative false rate `<= 0.005`;
- isolated LDHN hit rate `<= 0.03`;
- per-image negative false p95 `<= 0.05`;
- density-only matched control remains worse on recall/safety tradeoff.

Passing v2h-A authorizes only v2h-B shadow-modulation. It does not authorize F5, v3, D2, RARM, adapter training, or locked test.
