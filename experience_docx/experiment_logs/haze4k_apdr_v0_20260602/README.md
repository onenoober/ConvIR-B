# Haze4K APDR-ConvIR v0 Logs

Status: completed cloud stop20 scout; APDR-v0 failed the predeclared gate.

This directory contains text evidence for:

```text
APDR-v0 = official ConvIR-B anchor
        + RGB haze-prior encoder
        + bounded output residual adapters
        + spatial confidence gates
        + adapter-only training
```

Boundary:

- compare APDR-v0 against A0 official ConvIR-B Haze4K checkpoint;
- train only `APDR_*` parameters under `--apdr_train_scope apdr_only`;
- keep depth prior, diffusion/teacher, hard FFT, PFD/RHFD, HSCM, PFFB, and FAM
  modulation disabled;
- do not promote this exact APDR-v0 route after the failed stop20 gate.

Text artifacts:

| File | Purpose |
| --- | --- |
| `run_apdr_v0_stop20.sh` | Reproducible cloud command wrapper. |
| `preflight_apdr_v0.json` | Pair audit, checkpoint load, zero-init equivalence, finite backward, parameter count. |
| `train_ConvIR-Haze4K-APDR-v0-adapter-only-stop20-seed3407-20260602.log` | Corrected stop20 training log. |
| `train_ConvIR-Haze4K-APDR-v0-adapter-only-stop20-seed3407-20260602.crash_apdr_stats_padding_20260602_1747.log` | Archived first launch that crashed in APDR stats before the padding fix. |
| `scout_eval_compare_apdr_v0_stop20_seed3407_vs_a0.json` | A0-vs-APDR full-test summary. |
| `scout_eval_per_image_apdr_v0_stop20_seed3407_vs_a0.csv` | Per-image PSNR/SSIM deltas. |
| `scout_eval_bucket_analysis_apdr_v0_stop20_seed3407_vs_a0.json` | Hard/medium/easy bucket and regression distribution analysis. |
| `gate_apdr_v0_stop20_seed3407_vs_a0.json` | Predeclared stop20 gate result. |
| `status.txt` | Timestamped run status stream. |
| `tmux_apdr_v0_stop20.out` | Corrected run tmux transcript. |
| `tmux_apdr_v0_stop20.crash_apdr_stats_padding_20260602_1747.out` | Archived first launch tmux transcript. |

Text-only sync rule: keep checkpoints, image outputs, datasets, arrays, and raw
inference artifacts out of Git.

## Result Summary

Preflight passed:

- Haze4K pair audit: `3000/3000` train and `1000/1000` test pairs.
- Official ConvIR-B checkpoint loaded exactly into A0; APDR only missed
  expected `APDR_*` keys.
- Zero-init equivalence: random and real-batch `max_abs_diff = 0.0`.
- Trainable APDR params: `62,700`; frozen backbone params: `8,630,665`.

The first stop20 launch crashed only in APDR stats collection because raw
validation images were not factor-32 padded before `collect_apdr_stats`. The
fix padded stats inputs only; training loss, model, schedule, and gate were not
changed. The corrected cloud run completed stop20 and full-test evaluation.

Stop20 Best vs A0:

| Metric | Value | Gate |
| --- | ---: | --- |
| mean PSNR delta | `-0.0066543 dB` | fail |
| mean SSIM delta | `+0.0000197` | pass |
| hard bottom-25% delta | `-0.0009749 dB` | fail |
| easy top-25% delta | `-0.0150923 dB` | fail |
| strong-reference regressions | `100/250` | fail |
| severe regressions | `24/1000` | fail |
| worst10 mean delta | `-0.1759460 dB` | pass |
| median PSNR delta | `-0.0066500 dB` | fail |

Decision: `FAIL_STOP_APDR_V0_ADAPTER_ONLY`. The anchor-preserved APDR structure
is technically valid and active, but it does not improve hard cases and still
regresses easy/strong-reference samples beyond the stop20 gate. Keep this route
diagnostic-only; do not promote to 80/full training or add v1/v2 components on
top of this failed v0 result without a new mechanism change.
