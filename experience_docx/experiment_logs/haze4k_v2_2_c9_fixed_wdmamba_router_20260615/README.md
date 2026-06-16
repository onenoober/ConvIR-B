# Haze4K v2.2 C9 Fixed WDMamba Router Evidence

Decision: `LOCKED_WD0375_ONE_SHOT_PASS_REVIEW_DISTILLATION_LATER`

C9 used C8 train-derived per-image tables only. Locked test remained untouched.
The fixed `WD0375` profile passed C9-A, so C9-B router training was not run.
C9-C group-min shifted validation passed, and the sealed `WD0375` profile then
passed C10 formal 5x3 table replay.

The sealed locked one-shot was then consumed once for fixed `WD0375` only. The
command recorded `one_shot=true` and `no_tuning_from_locked=true`.

## Headline Metrics

| Stage | mean | hard bottom-25 | easy top-25 | positive | severe / 600 |
| --- | ---: | ---: | ---: | ---: | ---: |
| C9-A WD0375 full 600 | `+2.512202` | `+3.505615` | `+1.189484` | `0.973333` | `11.0` |
| C10 fold mean | `+2.516942` | `+3.523592` | `+1.213647` | `0.973556` | `10.942143` |
| C10 fold worst | `+2.311024` | `+3.347410` | `+0.857374` | `0.948276` | `21.818182` |
| Locked one-shot WD0375 | `+1.442090` | `+1.529767` | `+1.182529` | `0.938000` | `25.80` |

Worst C9-C bins still passed the fixed group gate: min mean `+1.124603`, min
hard `+1.552796`, min positive `0.900000`, and max severe `40/600`.

Primary outputs:

- `v22_c9a_fixed_profiles_summary.csv`
- `v22_c9a_fixed_profiles_groupmin_bins.csv`
- `v22_c9c_shifted_dimension_summary.csv`
- `v22_c9_decision.md`
- `v22_c9_summary.json`
- `v22_c10_formal_5x3_summary.csv`
- `v22_c10_formal_5x3_decision.md`
- `v22_locked_wd0375_one_shot_summary.json`
- `v22_locked_wd0375_one_shot_decision.md`
- `v22_locked_wd0375_one_shot_per_image.csv`

Locked output is evidence only. It must not tune alpha, features, checkpoints,
profiles, actions, experts, or distillation targets.
