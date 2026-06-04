# ConvIR-Dehaze-v1.3-HSDF

Date: 2026-06-04

Status: v1.3A and v1.3B intenal diagnostics completed; v1.3B failed the regular+hard gate; locked Haze4K test blocked.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: DPGA in-network depth/prior-guided adapter.
- Dataset or task: Haze4K train-derived intenal splits.
- Primary objective: recover hard-bottom samples without reviving easy/tail regressions.
- Execution environment: AutoDL `autodl-dehaze4`; local work is compile-only.
- Artifact root: `experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604/`.
- Branch or isolated workspace: `codex/haze4k-convir-v1-3-hard-selective-depth-fusion`.

## Baseline Contract

- Baseline implementation: ConvIR-B Haze4K base checkpoint, unchanged.
- Candidate initialization: ConvIR-B base checkpoint loaded into DPGA model with `DPGA_*` missing keys only.
- Training entrypoint: `Dehazing/ITS/main.py --arch dpga --mode train`.
- Evaluation entrypoint: `experience_docx/tools/eval_haze4k_checkpoint_compare.py`.
- Split policy: generate `train_inner`, `val_regular`, and `val_hard` before training; no locked Haze4K test in v1.3A.
- Checkpoint contract: select only on intenal regular+hard gate; Best/Final must both be reported.

## Most Valuable Attempt

- v1.1/v1.2 show DPGA safety is real but scale-only shallow DPGA has become easy-safe and hard-limited.
- The highest-value next variable is not more scale; it is the loss-mask definition that currently lets `high_anchor` protect almost the whole image.
- v1.3A tests whether hard gain is being suppressed by overbroad anchor preservation before adding a hard-gated bottleneck expert.
- Failure decides whether v1.3B should add the bottleneck hard expert; success avoids extra architecture capacity.

## Hypothesis

If `high_anchor` is allowed only on easy/strong-reference images while hard images preserve only sky/bright/color-risk regions, then hard-bottom improvement should rise on `val_hard` without losing v1.1-level regular/easy safety.

## Change

- Code branch: `codex/haze4k-convir-v1-3-hard-selective-depth-fusion`.
- v1.3A enabled mechanisms:
  - `--dpga_tc_mask_mode hard_selective`;
  - `--dpga_hard_sampler_json <v1.3 split json>`;
  - hard/medium/easy batch sampler;
  - `--dpga_hard_sample_lambda 0.20`, so hard samples receive the intended `1.2x` reconstruction weight;
  - `--dpga_hard_region_lambda 0.35`;
  - `--dpga_hard_region_error_threshold 0.010`, selected after the A0 mask audit showed hard-bucket anchor-error means around `0.0168`;
  - shallow-only DPGA adapter at scale `0.25`.
- v1.3B enabled mechanisms:
  - all v1.3A loss-mask, sampler, hard-sample, and hard-region settings;
  - `--dpga_active_adapters shallow,bottleneck`;
  - `--dpga_hard_gate_mode bottleneck`;
  - `--dpga_hard_gate_init_bias -3.0`;
  - `--dpga_bottleneck_scale_multiplier 2.0`, so the global `0.25` scale gives the bottleneck path a first-pass max near `0.05`;
  - `--dpga_hard_gate_lambda 0.01` with train-only A0 bucket supervision.
- Explicitly disabled mechanisms:
  - scale `0.75` or `1.0`;
  - locked Haze4K test;
  - APDR residual/action-bank revival;
  - full ConvIR finetune;
  - ungated bottleneck activation.
- Initialization: DPGA projection remains zero-init; ConvIR backbone remains frozen.

## Preflight

| Check | Pass line | Result |
| --- | --- | --- |
| Python compile | DPGA train/data/tools compile | pass locally on WSL `py310` |
| split generation | `train_inner`, `val_regular`, `val_hard`, A0 buckets written | pass; `intenal_val/haze4k_dpga_v13_regular_hard_seed3407.json` |
| tail mask audit | hard bucket no longer has legacy near-global protection under hard-selective mask | pass; `intermediates/dpga_v13_tail_mask_audit_by_bucket.csv` |
| hard proxy audit | deployable proxy AUC table written | pass; `intermediates/dpga_v13_hard_proxy_auc.csv` |
| no locked test | all commands use train-derived intenal splits | pass; locked Haze4K test not run |

## v1.3A Result

v1.3A completed on AutoDL `autodl-dehaze4` at `2026-06-04T16:26:48+08:00`.

| Gate item | Observed | Required | Result |
| --- | ---: | ---: | --- |
| `val_regular` mean delta | `+0.026333 dB` | `>= +0.035 dB` | fail |
| `val_regular` easy top-25 delta | `+0.023760 dB` | `>= 0` | pass |
| `val_regular` worst `<= -0.20 dB` | `4/300` | `<= 12/300` | pass |
| `val_regular` SSIM delta | `+0.000003` | `>= 0` | pass |
| `val_hard` hard bottom-25 delta | `+0.022099 dB` | `>= +0.040 dB` | fail |
| `val_hard` mean delta | `+0.024355 dB` | `>= 0` | pass |
| Final `val_regular` mean delta | `+0.018906 dB` | `>= 0` | pass |
| Final `val_hard` mean delta | `+0.016828 dB` | `>= 0` | pass |

Decision: v1.3A fixed the overbroad mask mechanism without tail collapse, but hard gain stayed below the written gate. This authorizes v1.3B hard-gated bottleneck capacity. It does not authorize locked Haze4K test.

## v1.3B Result

v1.3B completed on AutoDL `autodl-dehaze4` at `2026-06-04T17:25:32+08:00` after one early engineering stop.

The first v1.3B launch at `2026-06-04T16:35:27+08:00` was stopped because the anchor-forward path cleared `_last_hard_gate`, so `dpga_hard_gate_bce` was not entering the loss. The fixed run restarted at `2026-06-04T16:48:16+08:00`; early logs showed `hard_gate` about `0.048` and nonzero `hard_gate_bce`, and training completed with `train_rc=0`.

| Gate item | Observed | Required | Result |
| --- | ---: | ---: | --- |
| `val_regular` mean delta | `+0.025839 dB` | `>= +0.040 dB` | fail |
| `val_regular` easy top-25 delta | `+0.022205 dB` | `>= 0` | pass |
| `val_regular` worst `<= -0.20 dB` | `3/300` | `<= 12/300` | pass |
| `val_regular` positive ratio | `0.586667` | `>= 0.62` | fail |
| `val_regular` strong regression ratio | `0.200000` | `<= 0.16` | fail |
| `val_regular` SSIM delta | `+0.000001` | `>= 0` | pass |
| `val_hard` hard bottom-25 delta | `+0.023642 dB` | `>= +0.050 dB` | fail |
| `val_hard` mean delta | `+0.024686 dB` | `>= 0` | pass |
| Final `val_regular` mean delta | `+0.025573 dB` | `>= 0` | pass |
| Final `val_hard` mean delta | `+0.024423 dB` | `>= 0` | pass |

Decision: v1.3B remains safe-ish on intenal splits but does not improve hard recovery beyond v1.3A and fails the written pass line. It does not authorize locked Haze4K test. Treat current HSDF hard-gated bottleneck as a completed diagnostic, not a promotion route.

Corrected runtime ablation at the route scale `0.25` shows the bottleneck path added very little: Best `all_adapters` mean/hard/easy deltas were `+0.025839 / +0.020272 / +0.022205 dB`, while `no_bottleneck` was `+0.025112 / +0.018995 / +0.022434 dB` and `bottleneck_only` was only `+0.000824 / +0.001295 / -0.000103 dB`. The earlier runtime-ablation outputs that accidentally used module scale `1.0` are archived under `.scale1_bug_20260604T1725`.

## Mechanism Metrics

| Metric | Why it matches the route | Gate subset | Final artifact |
| --- | --- | --- | --- |
| mask ratio by bucket | verifies the overbroad anchor-protect diagnosis | train-derived regular/hard split | `dpga_v13_tail_mask_audit_by_bucket.csv` |
| A0 PSNR bucket balance | proves `val_hard` is actually hard | split JSON rows | `dpga_v13_val_split_audit.csv` |
| hard proxy AUC | checks deployable hard-gate features before v1.3B | train rows | `dpga_v13_hard_proxy_auc.csv` |
| regular+hard gate | prevents easy-safe configs from being selected as hard gains | Best/Final on both splits | `dpga_v13_gate_eval_regular_and_hard.json` |
| corrected runtime ablation | checks whether shallow or bottleneck supplies the gain at route scale | `val_regular` only | `runtime_diagnostics_v13b_val_regular/dpga_v13b_runtime_ablation_on_val_inner.csv` |

## Gates

### v1.3A Pass Line

- `val_regular` mean PSNR delta `>= +0.035 dB`.
- `val_regular` easy top-25 delta `>= 0`.
- `val_regular` worst `<= 12/300`.
- `val_hard` hard bottom-25 delta `>= +0.040 dB`.
- SSIM delta `>= 0`.

### Promotion

v1.3A is diagnostic only. If it passes and hard gain is near `+0.04 dB`, do not add bottleneck capacity yet. If hard remains below gate while safety holds, authorize v1.3B hard-gated bottleneck expert. Only a future v1.3B regular+hard pass can authorize one locked Haze4K test.

### v1.3B Pass Line

- `val_regular` mean PSNR delta `>= +0.040 dB`.
- `val_regular` easy top-25 delta `>= 0`.
- `val_regular` worst `<= 12/300` or no worse than v1.1.
- `val_regular` strong regression ratio `<= 0.16`.
- `val_hard` hard bottom-25 delta `>= +0.050 dB`.
- Positive ratio `>= 0.62`.
- SSIM delta `>= 0`.
