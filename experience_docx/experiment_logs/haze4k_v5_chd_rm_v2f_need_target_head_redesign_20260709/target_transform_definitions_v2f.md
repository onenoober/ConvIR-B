# v2f Target Transform Definitions

- `global`: current v2e global quantile target.
- `density_conditioned_q`: raw need CDF fitted inside train_inner density bins.
- `excess_over_density_q`: raw need minus train_inner density-bin mean, then CDF fitted inside density bins.
- `core_ignore`: positive core is target >= q80 in low density and stable-high; negative core is target <= q33 in low density; other pixels are ignored by future head canaries.

All transforms are fitted on train_inner and evaluated on val_inner. Locked Haze4K test is not used.
