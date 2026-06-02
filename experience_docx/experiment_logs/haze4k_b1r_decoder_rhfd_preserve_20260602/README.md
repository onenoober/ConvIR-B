# Haze4K B1r Decoder RHFD Preserve Logs

Status: script-ready; run artifacts pending cloud execution.

This directory is reserved for the B1r rescue route:

```text
B1r = decoder-side RHFD-Lite + adapter-only preservation training
```

Boundary:

- compare against A0 official ConvIR-B Haze4K checkpoint, not A1 stop20;
- train only `PFD_DECODER_RHFD*` under `--pfd_adapter_only 1`;
- keep `--pfd_rhfd 0`, `--pfd_hscm 0`, `--pfd_pffb 0`, and `--pfd_teacher 0`;
- do not launch B2/B3 or any HSCM/PFFB/hard-frequency/haze-prior variants from this route.

Planned text artifacts:

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

Text-only sync rule: keep checkpoints, images, datasets, arrays, and raw inference outputs out of Git.
