# Haze4K ConvIR-Dehaze-v1.1-DPGA-Tail-Control

Date: 2026-06-04

Status: completed diagnostic route; v1.1 and v1.2 `val_inner` gates failed on
hard bottom-25% gain, so locked Haze4K test remains blocked.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: DPGA in-network depth/prior-guided adapter.
- Primary objective: control DPGA-Lite strong-reference and sky/airlight/color
  tail risk before launching a new training run.
- Main metric: delta against official frozen ConvIR-B A0 on mean PSNR, hard
  bottom-25%, easy top-25%, strong-reference regressions, worst `<= -0.20 dB`
  regressions, and bright/sky proxy masked deltas.
- Execution environment: cloud only for diagnostics/training,
  `autodl-dehaze4`; local work is compile/static checks only.
- Branch or isolated workspace:
  `codex/haze4k-convir-v1-1-dpga-tail-control`.

## Baseline Contract

- Baseline checkpoint:
  `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`.
- Prior route checkpoints:
  `/root/autodl-tmp/workspace/ConvIR-B-dpga-lite-826caaf/Dehazing/ITS/results/ConvIR-Haze4K-DPGA-Lite-v1.0-adapter-only-stop20-seed3407-20260604/Training-Results`.
- Dataset:
  `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`.
- Depth cache:
  `/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf`.
- Evaluation script:
  `experience_docx/tools/audit_haze4k_dpga_runtime_variants.py`.

## Route Boundary

Allowed now:

- Evaluate existing DPGA-Lite Best/Final checkpoints without retraining.
- Runtime-close DPGA adapters by insertion point.
- Runtime-scale effective DPGA modulation.
- Generate a fixed internal train/validation split for future checkpoint
  selection.

Not allowed now:

- No APDR output residual revival.
- No ConvIR backbone finetune.
- No token-wise low-rank model jump.
- No new test-selected Best claim.

## Preflight Diagnostics

| Diagnostic | Artifact | Pass/use line |
| --- | --- | --- |
| Module ablation | `dpga_module_ablation_best_final.csv` | identify whether shallow, bottleneck, or skip drives gain/tail |
| Scale sweep | `dpga_scale_sweep_best_final.csv` | find whether `0.5` or `0.75` retains positive mean with fewer strong/worst regressions |
| Internal val split | `haze4k_train_inner_val_inner_seed3407.json` | future v1.1 Best must be selected on `val_inner`, not test |

## Scout Gate For Next Training

Start DPGA-v1.1 training only after the runtime diagnostics choose a concrete
configuration. The first candidate should prefer fewer active adapters and a
smaller scale cap if those reduce strong-reference regressions without erasing
mean gain.

Promotion from future v1.1 training requires:

- `val_inner` mean delta `>= +0.03 dB`;
- `val_inner` hard bottom-25% delta `>= +0.03 dB`;
- `val_inner` easy top-25% delta `>= 0`;
- strong-reference regressions lower than DPGA-Lite Best;
- locked test eval only after checkpoint/config/scale selection.

## Immediate Commands

On `autodl-dehaze4`, after the branch/workspace is synced:

```bash
cd /root/autodl-tmp/workspace/ConvIR-B-dpga-tail-control
bash experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604/run_make_internal_val_split.sh
tmux new -d -s dpga_tail_diag 'cd /root/autodl-tmp/workspace/ConvIR-B-dpga-tail-control && bash experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604/run_dpga_runtime_diagnostics.sh'
```

## Cloud Results

The runtime diagnostics selected a conservative shallow-only path and scale
control before architectural expansion.

| Step | Config | Gate result | Main observation |
| --- | --- | --- | --- |
| DPGA-v1.1 | shallow-only, scale `0.25`, anchor `0.08` | fail | Best mean `+0.037036 dB` passed, but hard bottom-25% `+0.023367 dB` missed the `+0.030 dB` gate. |
| DPGA-v1.2 | shallow-only, scale `0.5`, anchor `0.04` | fail | Best mean improved to `+0.042656 dB`, but hard bottom-25% only reached `+0.026225 dB`; worst `<= -0.20 dB` regressions rose to `16/300`. |

Both runs preserved positive mean movement and avoided a broad collapse, but
neither delivered enough hard-case gain for a locked Haze4K test. The v1.2
failure analysis explicitly says not to launch a higher-scale follow-up without
a new diagnostic.

## Decision

`STOP_DPGA_SCALE_ONLY_TAIL_CONTROL`. Keep this route as diagnostic evidence.
Do not run locked test for v1.1 or v1.2. A future DPGA route needs a new
mechanism or diagnostic that improves hard bottom-25% gain without increasing
worst-tail regressions.
