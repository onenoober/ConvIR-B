# APDR-v0.3 Delta Learnability Overfit

Date: 2026-06-03

Status: completed diagnostic; gate failed.

## Command

```bash
bash experience_docx/experiment_logs/haze4k_apdr_v0_3_delta_learnability_20260603/run_apdr_v0_3_delta_learnability_32.sh
```

Executed on AutoDL `autodl-dehaze3` under:

```text
/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-3-shed-diagnostics
```

## Setup

- Train subset: first 32 Haze4K train images, full image, no random crop.
- Frozen model: APDR-v0.2RC selector checkpoint from the frozen residual run.
- Trainable parameters: full-scale APDR residual body/head only.
- Loss: masked delta L1, `|M_safe * (residual_raw - Delta_star)|`.
- Steps: `400`.
- Learning rate: `1e-3`.
- Grad clip: `1.0`.

## Result

| Check | Observed | Required | Result |
| --- | ---: | ---: | --- |
| weighted delta L1 drop | `0.0231` | `>= 0.30` | fail |
| oracle gain recovery | `0.0111` | `>= 0.50` | fail |
| residual/delta correlation | `0.1552` | `>= 0.40` | fail |
| hard train PSNR gain | `+0.0240 dB` | `>= +0.30 dB` | fail |

## Decision

`FAIL_STOP_CURRENT_APDR_RESIDUAL_LEARNABILITY`.

Do not continue the current tiny residual head over frozen selector context.
The next useful diagnostic is residual-source oracle ablation, followed by a
new residual expression path only if the ablation identifies a compact source
such as low-frequency veil or color affine correction.
