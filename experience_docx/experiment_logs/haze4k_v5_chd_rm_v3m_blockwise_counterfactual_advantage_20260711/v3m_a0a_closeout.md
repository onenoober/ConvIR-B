# v3m A0a Common-Action Granularity Closeout

Date: 2026-07-11

Decision:
`V3M_A0_COMMON_ACTION_GRANULARITY_PASS_AUTHORIZE_A0B_DENSE_AND_CONTINUOUS_MECHANISM_ONLY`.

## Scope

A0a used the frozen v3l `D_ref` and `D_rep` direct operators on the fixed
train-derived clean-reference OOF groups. Image, block32, block16, and pixel
grid oracles all used exactly `{0, 0.125, 0.25, 0.5, 1.0}`. Fixed
`alpha=0.125` on the same rows was the reference. Route-confirm was emitted
for audit only and was not used to select the ladder, a policy, a threshold, or
the gate.

## Dual-Operator Gate

| Operator | Mean lift (dB) | Lift CI95 low (dB) | Pixel-grid lift (dB) | Block16 retention | Retention CI95 low | Candidate p10 (dB) | Reference p10 (dB) | Severe candidate / reference | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `D_ref` | `+0.3571825` | `+0.3381417` | `+0.4211347` | `0.8481430` | `0.8403679` | `+0.0080694` | `-0.0274989` | `0 / 0` | pass |
| `D_rep` | `+0.3555132` | `+0.3374088` | `+0.4182782` | `0.8499444` | `0.8423180` | `+0.0068382` | `-0.0269539` | `0 / 0` | pass |

Both lower confidence bounds exceeded the preregistered `0.80` retention
threshold and zero, respectively. Both p10 and worst preservation checks
passed, and neither candidate added a severe regression.

## Integrity And Repair Record

The first frozen replay completed inference and wrote raw cloud-only tables but
failed only during bootstrap summary construction. The first recovery rebuilt
compact summaries without inference. Its operator-agreement diagnostic was
`NaN` because it read `mean_selected_alpha_mean`; raw policy rows contain
`selected_alpha_mean`.

The second recovery was constrained to `--summarize_existing_raw` and backed up
the prior summary JSON and agreement CSV. It did not load a model or checkpoint,
run inference, modify raw data, touch locked test, authorize canary, or train.
The following raw artifacts were unchanged before and after the repair:

| Raw artifact | Lines including header | SHA256 |
| --- | ---: | --- |
| full policy rows | `43,201` | `b8455838ab2d703c44d425f2739f4ea4c2d6aa8efdb66960c5733a41781c690e` |
| OOF policy rows | `28,801` | `b4a10184fab77b0045440dc88530d000a892acf2105a6295d5ad8a488c67ecb1` |
| policy summary | `49` | `5c7bb0b0832e50e40fc6ef6ee02bc4539ce8b56bb59649ec3ac163b955380273` |

Both A0a gate rows matched exactly before and after the compact rebuild. The
repaired cross-operator agreement on the 1,200 paired OOF clean-reference
groups was finite for every common oracle:

| Policy | Pearson on selected alpha mean | Mean absolute difference |
| --- | ---: | ---: |
| image grid | `0.9688461` | `0.0296875` |
| block16 grid | `0.9975927` | `0.0092068` |
| block32 grid | `0.9944302` | `0.0151715` |
| pixel grid | `0.9995011` | `0.0030044` |

## Authorization Boundary

A0a authorizes only A0b dense-grid and continuous-pixel mechanism cross-audit.
It does not authorize A1 feasible local actuation, B0 full-physics audit, B1
privileged policy, B2 proxy, controller training, canary, locked-test access,
or any new model search.
