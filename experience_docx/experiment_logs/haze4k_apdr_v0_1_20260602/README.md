# Haze4K APDR-ConvIR v0.1 Logs

Status: completed cloud stop20 scout; APDR-v0.1 failed the hard-gain/mean gate.

This directory contains text evidence for the APDR-v0.1 anchor-risk route.

```text
APDR-v0.1 = official ConvIR-B anchor
          + RGB haze-prior encoder
          + bounded full-output residual adapter
          + spatial confidence gate
          + adapter-only full-branch training
          + A0-risk anchor/no-degrade loss
          + direct gate supervision from A0 error
```

Boundary:

- compare APDR-v0.1 against the official ConvIR-B Haze4K checkpoint;
- train only the active APDR full branch under `--apdr_train_scope apdr_only`;
- keep depth prior, diffusion/teacher, hard FFT, PFD/RHFD, HSCM, PFFB, FAM,
  and low-frequency veil disabled;
- do not promote this exact route unless the stop20 gate passes.

Text artifacts:

| File | Purpose |
| --- | --- |
| `run_apdr_v0_1_stop20.sh` | Reproducible cloud command wrapper. |
| `preflight_apdr_v0_1.json` | Pair audit, checkpoint load, zero-init equivalence, finite backward, parameter count. |
| `train_ConvIR-Haze4K-APDR-v0.1-anchor-risk-stop20-seed3407-20260602.log` | Stop20 training log. |
| `scout_eval_compare_apdr_v0_1_stop20_seed3407_vs_a0.json` | A0-vs-APDR full-test summary. |
| `scout_eval_per_image_apdr_v0_1_stop20_seed3407_vs_a0.csv` | Per-image PSNR/SSIM deltas. |
| `scout_eval_bucket_analysis_apdr_v0_1_stop20_seed3407_vs_a0.json` | Hard/medium/easy bucket and regression distribution analysis. |
| `gate_apdr_v0_1_stop20_seed3407_vs_a0.json` | Predeclared stop20 gate result. |
| `status.txt` | Timestamped run status stream. |
| `launcher.out` | Detached cloud launcher transcript including preflight, bucket analysis, and gate output. |

Text-only sync rule: keep checkpoints, image outputs, datasets, arrays, and raw
inference artifacts out of Git.

## Result Summary

Cloud execution ran on AutoDL `autodl-dehaze3` in
`/root/autodl-tmp/workspace/ConvIR-B-apdr-convir-v0-1`.

Preflight passed:

- Haze4K pair audit: `3000/3000` train and `1000/1000` test pairs.
- Official ConvIR-B checkpoint loaded exactly into A0; APDR only missed
  expected `APDR_*` keys.
- Zero-init equivalence: random and real-batch `max_abs_diff = 0.0`.
- Active trainable APDR params: `19,876`; frozen params: `8,673,489`.
- Finite backward under the v0.1 loss was valid; direct gate supervision gave
  nonzero gradients to the full gate head at initialization.

Stop20 Best vs A0:

| Metric | Value | Gate |
| --- | ---: | --- |
| mean PSNR delta | `+0.0001148 dB` | fail |
| mean SSIM delta | `+0.00000335` | pass |
| median PSNR delta | `+0.0003586 dB` | pass |
| hard bottom-25% delta | `+0.0006680 dB` | fail |
| easy top-25% delta | `-0.0010661 dB` | pass |
| strong-reference regressions | `1/250` | pass |
| severe regressions | `0/1000` | pass |
| true worst-10-image mean delta | `-0.0449356 dB` | pass |

Decision: `FAIL_STOP_APDR_V0_1_ANCHOR_RISK_HARD_GAIN`. The A0-risk/no-degrade
training constraint fixed the main v0 preservation failure, but it did not
create the required hard-sample gain. Keep v0.1 diagnostic-only; do not promote
to 80/full. The next APDR attempt should not simply increase residual strength
unless it adds a stronger hard-case selector.
