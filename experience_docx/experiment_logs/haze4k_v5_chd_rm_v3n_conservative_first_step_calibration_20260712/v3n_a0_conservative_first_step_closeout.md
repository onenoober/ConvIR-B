# v3n A0 Conservative First-Step Label Preflight

Decision: `V3N_A0_CONSERVATIVE_LABEL_PREFLIGHT_FAIL_STOP_NO_REPLAY`.

This is a label-only diagnostic over the v3m-A1 cloud block table. It uses a
fixed rule: default to `alpha=0.125`, and allow only `alpha=0.25` when
`direct_step_energy` exceeds the 99th percentile of train-fold negative blocks.
It does not train, tune thresholds, replay images, use route-confirm, touch
canary, or touch locked test.

| Operator | Selected coverage | Positive recall | Negative false rate | Selected precision | Selected over-escalate | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `D_ref` | `0.0000000` | `0.0000000` | `0.0000000` | `nan` | `nan` | `False` |
| `D_rep` | `0.0000000` | `0.0000000` | `0.0000000` | `nan` | `nan` | `False` |

No policy utility is claimed by this label-only phase. A pass authorizes only a
separate A1 32-image replay-smoke preflight; a fail stops this v3n rule.
