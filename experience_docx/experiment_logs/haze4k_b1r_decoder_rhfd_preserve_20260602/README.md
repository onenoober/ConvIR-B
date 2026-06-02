# Haze4K B1r Decoder RHFD Preserve Logs

Status: completed; B1r adapter-only stop20 failed the predeclared B1r gate.

This directory is reserved for the B1r rescue route:

```text
B1r = decoder-side RHFD-Lite + adapter-only preservation training
```

Boundary:

- compare against A0 official ConvIR-B Haze4K checkpoint, not A1 stop20;
- train only `PFD_DECODER_RHFD*` under `--pfd_adapter_only 1`;
- keep `--pfd_rhfd 0`, `--pfd_hscm 0`, `--pfd_pffb 0`, and `--pfd_teacher 0`;
- do not launch B2/B3 or any HSCM/PFFB/hard-frequency/haze-prior variants from this route.

Text artifacts:

| File | Purpose |
| --- | --- |
| `run_b1r_decoder_rhfd_preserve.sh` | Reproducible cloud command wrapper. |
| `preflight_b1r_decoder_rhfd.json` | Pair audit, checkpoint load, zero-init equivalence, trainable adapter count. |
| `B1r_adapter_only_stop10_seed3407.log` | Stop10 adapter-only training log. |
| `B1r_adapter_only_stop20_seed3407.log` | Stop20 adapter-only training log if stop10 remains plausible. |
| `scout_eval_compare_seed3407_B1r_stop10_vs_A0_best.json` | Stop10 A0-vs-B1r summary. |
| `scout_eval_per_image_seed3407_B1r_stop10_vs_A0_best.csv` | Stop10 per-image deltas. |
| `scout_eval_bucket_analysis_seed3407_B1r_stop10_vs_A0_best.json` | Stop10 hard/easy bucket analysis. |
| `scout_eval_compare_seed3407_B1r_stop20_vs_A0_best.json` | Stop20 A0-vs-B1r summary. |
| `scout_eval_per_image_seed3407_B1r_stop20_vs_A0_best.csv` | Stop20 per-image deltas. |
| `scout_eval_bucket_analysis_seed3407_B1r_stop20_vs_A0_best.json` | Stop20 hard/easy bucket analysis. |
| `gate_B1r_stop20.json` | Stop20 preservation gate result. |
| `status.txt` | Timestamped run status. |
| `tmux.out` | Detached tmux transcript. |

Text-only sync rule: keep checkpoints, images, datasets, arrays, and raw inference outputs out of Git.

## Result Summary

Preflight passed with zero-init equivalence (`max_abs_diff = 0.0`) and exactly
`3712` trainable B1r adapter parameters.

Stop20 Best vs A0:

| Metric | Value | Gate |
| --- | ---: | --- |
| global mean PSNR delta | `+0.0028 dB` | pass |
| mean SSIM delta | `+0.000050` | pass |
| hard bottom-25% delta | `+0.0461 dB` | fail |
| easy top-25% delta | `-0.0248 dB` | pass |
| strong-reference regressions | `103/250` | fail |
| severe regressions | `57/1000` | pass |

Decision: `FAIL_STOP_B1R_DECODER_RHFD_ADAPTER_ONLY`. The route is much more
preservation-stable than B1 feature-delta RHFD, but the hard-case gain is too
small and strong-reference regressions remain above the gate.
