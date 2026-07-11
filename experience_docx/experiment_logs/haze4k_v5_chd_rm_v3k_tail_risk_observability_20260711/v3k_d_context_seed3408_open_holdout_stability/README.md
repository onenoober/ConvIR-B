# v3k-D Context Seed Stability

This is a lightweight robustness check for the v3k-C open-holdout result.
It retrains only the context diagnostic head with a different seed and evaluates
the OOF-preselected alpha values 0.125 and 0.25 on historical `val_inner`.

This remains open-holdout diagnostic evidence only. It is not a new sealed split
and cannot authorize canary or promotion.
