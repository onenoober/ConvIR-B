# APDR-v0.3 SHED Safe Expert Replay

Date: 2026-06-03

Status: completed diagnostic; all replay candidates failed gate.

## Command

```bash
bash experience_docx/experiment_logs/haze4k_apdr_v0_3_shed_replay_20260603/run_apdr_v0_3_shed_replay.sh
```

Executed on AutoDL `autodl-dehaze3` under:

```text
/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-3-shed-diagnostics
```

## Result

| Expert | Gate | Safe mean | Safe hard bottom-25% | Safe easy top-25% | Strong regressions | Severe regressions | Raw expert mean vs A0 anchor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FAM2-only | fail | `-0.1259` | `-0.1758` | `-0.0003` | `1/250` | `204/1000` | `-9.3243` |
| FAM2 confidence-gated | fail | `-0.1365` | `-0.1913` | `+0.0004` | `2/250` | `203/1000` | `-9.0459` |
| HardFreq | fail | `-0.1156` | `-0.1688` | `+0.0002` | `2/250` | `205/1000` | `-9.7109` |
| PFD B1 | fail | `+0.0006` | `+0.0071` | `+0.0000` | `0/250` | `48/1000` | `-0.8031` |
| Haze-prior SCM | fail | `-0.1106` | `-0.1498` | `+0.0015` | `2/250` | `210/1000` | `-9.5836` |

## Interpretation

The v0.2RC action mask does not make historical hard-positive checkpoints
usable as direct deltas against the official ConvIR-B A0 anchor. FAM2,
HardFreq, and haze-prior SCM were hard-positive relative to their matched
stop20 baselines, not relative to the official A0 anchor. PFD B1 is closer but
still lacks useful hard gain after masking.

## Decision

`FAIL_STOP_APDR_V0_3_SHED_DIRECT_REPLAY`.

Do not launch a direct safe-expert APDR stop20 scout from these checkpoints.
Use this evidence only to motivate a teacher/source ablation if a future
residual branch is redesigned.
