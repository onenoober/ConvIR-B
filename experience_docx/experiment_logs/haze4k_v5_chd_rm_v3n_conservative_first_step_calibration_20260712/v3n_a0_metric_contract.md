# v3n A0 Metric Contract

Decision before launch:
`V3N_START_A0_CONSERVATIVE_FIRST_STEP_LABEL_PREFLIGHT_ONLY`.

## Inputs

- Source rows: v3m A1 cloud-only block table
  `v3m_a1_block_rows_cloud_only.csv`.
- Operators: `D_ref`, `D_rep`.
- Split: v3m train-derived 1,200-image OOF folds only.
- Score: `direct_step_energy`.

## Fixed Rule

For each operator and held-out fold:

- train negatives: `oracle_alpha <= 0.125` in the other four folds;
- threshold: 99th percentile of train negative `direct_step_energy`;
- held-out selected action: `alpha=0.25` if score is above threshold;
- default action: `alpha=0.125`.

No other threshold, score, action, quantile, fold definition, or feature family
may be selected from A0 results.

## Gates

Both operators must pass all:

- held-out negative false rate `<= 0.0125`;
- max per-fold negative false rate `<= 0.02`;
- selected coverage `>= 0.005`;
- min per-fold selected coverage `>= 0.0025`;
- selected precision `>= 0.60`;
- positive recall `>= 0.01`.

## Authorized Output

Pass authorizes only a separate A1 32-image replay smoke. A0 cannot authorize
formal replay, route-confirm, canary, locked test, training, learned ranker,
physics/proxy continuation, or deployment.
