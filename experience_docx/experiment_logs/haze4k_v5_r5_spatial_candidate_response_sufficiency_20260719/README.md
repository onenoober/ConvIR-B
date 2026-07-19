# Haze4K v5 R5 Spatial Candidate-Response Sufficiency Evidence

Date: 2026-07-19

Formal terminal:
`COMPLETED_GATE_FAIL / R5_A0_SPATIAL_RESPONSE_FUTILITY_OR_SAFETY_FAIL_STOP / NONE`.

The authoritative run is `r5-a0-spatial-response-screen-r2` at route commit
`7e75eed504b2ead65a1971ec250dc7f59a79574d`. The typed closeout SHA-256 is
`e8d6151a9d7fc1198db1db1a0fc44e2de91da0bea0f4a01053c68bf0d5c0e4e7`.
The route card is
`../../experiment_cards/2026-07-19-haze4k-v5-r5-spatial-candidate-response-sufficiency.md`;
the central index is `../../EXPERIMENT_INDEX.md`; and machine terminal
authority is `../../EXPERIMENT_TERMINAL_INDEX.jsonl`.

## Reading Order

1. Read `r5_a0_spatial_response_sufficiency_closeout.json` for terminal
   authority and identity-bound evidence hashes.
2. Read `r5_a0_scientific_conclusion.json` for the single route conclusion.
3. Read `r5_a0_gate_summary.json`, `r5_a0_bootstrap_summary.json` and
   `r5_a0_structural_summary.json` before interpreting individual metrics.
4. Use the remaining compact JSON/CSV files for preregistered mechanism,
   calibration, stability and safety interpretation.

## Evidence Roles

- R5-A0 r2 is a preregistered `development_screening` operation and is the
  formal route-decision evidence for R5. It is not independent external
  validation.
- R4B-A1's formal terminal and raw-support reproduction are historical inputs.
  Its abstention, fold-coverage and candidate-risk decompositions remain
  post-hoc exploratory evidence and do not become R5 endpoints.
- R5-A0 r1 is engineering/debug evidence only. It completed the frozen feature
  and training units but failed while finalization traversed non-evaluated
  folds. It remains `FAILED_ENGINEERING / null / NONE`; none of its partial
  CSVs is scientific evidence.

## Formal Result

All structural checks pass: 1,536 feature units, 16 seed-model training units,
384 evaluated development groups, folds 0/1, exact fixed coverage, complete
cells and rows, finite metrics, exact role isolation, valid shuffle controls,
and no protected-data access.

At the fixed primary coverage, the true spatial response cell achieves:

| Estimand | Point | LCB95 | UCB95 | Reading |
| --- | ---: | ---: | ---: | --- |
| Worse-operator population gain | `+0.025941 dB` | `+0.012476` | `+0.040757` | mean non-futility passes |
| Oracle retention | `0.172387` | `0.083632` | `0.264425` | screen upper-bound target reached |
| Spatial minus pooled | `+0.020161 dB` | `+0.006707` | `+0.034034` | spatial layout adds average value |
| True minus shuffle | `+0.020303 dB` | `+0.005851` | `+0.035699` | layout specificity is supported |
| Spatial minus generic | `+0.007913 dB` | `-0.008824` | `+0.025019` | direction positive, incremental specificity unresolved |
| Severe AUROC | `0.840579` | `0.804543` | `0.875661` | risk is identifiable |
| Severe AUPRC lift | `0.386462` | `0.314581` | `0.465035` | risk lift passes |

The same primary policy fails the safety contract: it selects 10 severe and 3
hard groups; the all-group severe exact UCB95 is `0.043772` against `0.010`;
CVaR5 spatial-minus-pooled is `-0.174216 dB` with LCB95 `-0.302166 dB` against
the `-0.005 dB` floor; and protected-cell harm increment is `+0.016300 dB`
with UCB95 `+0.030836 dB` against the `+0.005 dB` ceiling.

## Mechanism Conclusion

Spatial candidate-relative response is informative, so the broad claim that
the available representation contains no material signed action information is
too strong. But the frozen representation/readout/coverage contract cannot
turn that information into safe coverage: average gains coexist with severe,
hard, CVaR5 and protected-cell failures. The bottleneck is narrowed to locally
safe calibration and action/abstention execution, not generic risk detection or
the mere absence of spatial layout.

This finding does not authorize a nearby architecture or hyperparameter search.
The typed FAIL closes R5 and blocks R5-A1, confirmation, canary and locked test.

## Artifact Boundary

GitHub retains this README, the route conclusion, typed closeout and the 16
closeout-bound compact result files. Cloud retains raw candidate tensors,
per-image/action/operator/cell rows, per-seed prediction vectors, model states,
runtime logs, datasets, labels, images and arrays. No confirmation outcome,
historical A1X outcome, canary or locked-test datum was accessed.
