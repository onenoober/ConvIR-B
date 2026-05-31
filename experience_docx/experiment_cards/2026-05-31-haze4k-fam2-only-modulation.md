# Haze4K FAM2-Only Feature Modulation

Date: 2026-05-31

Status: 20-epoch gate complete; current unbounded FAM2-only is diagnostic,
not promoted to full or repeat seed.

## Scope

- Project: ConvIR-B dehazing.
- Model family: ConvIR-B.
- Dataset or task: Haze4K image dehazing.
- Primary objective: test whether half-scale FAM2-only conditional modulation preserves the useful signal from the FAM route while avoiding the strong-case regressions seen in FAM1+FAM2.
- Main metric: PSNR.
- Secondary metrics: SSIM, per-image PSNR delta, strong-case regression count, gamma/beta activation statistics, latency, peak GPU memory.
- Execution environment: `autodl-dehaze3`, RTX 5090, PyTorch `2.11.0+cu128`.
- Artifact root: `Dehazing/ITS/results/ConvIR-Haze4K-fam2-modres-*`.
- Branch or isolated workspace: `codex/haze4k-fam2-only`, based on `codex/haze4k-repro`.
- Review package location: `experience_docx/experiment_logs/haze4k_fam2_modres_*`.

## Baseline Contract

- Official pretrained baseline: ConvIR-B Haze4K checkpoint `haze4k-base.pkl`, local reproduction `34.14 dB PSNR / 0.98971 SSIM`, official table `34.15 / 0.99`.
- Official ConvIR-B size: `8.63M` parameters, `71.22G` FLOPs.
- Training comparison baseline: matched from-scratch `--fam_mode original` with the same seed, data, batch, schedule horizon, stop epoch, validation frequency, and hardware as FAM2-only.
- Baseline implementation: `Dehazing/ITS/main.py --version base --data Haze4K --fam_mode original`.
- Candidate implementation: `Dehazing/ITS/main.py --version base --data Haze4K --fam_mode fam2_modres`.
- Dataset and split: `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`; train has 3000 image pairs after extension filtering; test/validation has 1000 image pairs.
- Metric implementation: training validation PSNR from `Dehazing/ITS/valid.py`; full-test PSNR/SSIM/delta from `experience_docx/tools/eval_haze4k_checkpoint_compare.py`.
- Scheduler policy: keep the official Haze4K horizon as `--num_epoch 1000`; use `--stop_epoch` for fixed-budget gates.
- Reference entrypoints that must remain stable: default `--fam_mode original` must strict-load official checkpoints and reproduce the pretrained baseline.

## Most Valuable Attempt

- Why this is the highest-value next attempt: the earlier FAM1+FAM2 scout showed active modulation and a small mean PSNR gain, but failed preservation with many strong-case regressions. FAM2-only isolates the half-scale fusion point that is cheaper and less globally intrusive.
- Target failure or opportunity: non-homogeneous haze may benefit from regional half-scale condition features without allowing the lower-resolution FAM1 branch to shift global color or structure.
- Cheap preflight evidence: FAM2-only zero initialization must be numerically equivalent to original FAM; FAM2 gamma/beta must start at zero and become non-zero only through training.
- Earliest decisive gate: zero-init equivalence plus finite one-batch train probe.
- Expected cost or attempt-count saving: reject or narrow FAM before spending on FAM1+FAM2 or FFL.
- What success decides: run FAM2-only longer/full and repeat the seed before revisiting FAM1.
- What failure decides: pause FAM and inspect implementation/eval equivalence before moving to FFL.

## Hypothesis

- Observed failure: FAM1+FAM2 had a small average PSNR improvement but negative median delta and large strong-reference regression count.
- Target mechanism: restrict conditional modulation to FAM2 at 1/2 scale.
- Primary variable: FAM insertion location.

Mechanism sentence:

```text
If zero-initialized SCM-conditioned modulation is applied only at FAM2,
Haze4K hard or spatially non-homogeneous samples should improve while
strong-reference samples avoid the broad regressions seen when FAM1 is also
modulated.
```

## Change

- Exact code/config change: add `--fam_mode fam2_modres`; FAM2 uses zero-initialized `1x1` gamma/beta modulation, FAM1 remains original.
- Enabled mechanisms: FAM2-only SCM-conditioned residual feature modulation.
- Explicitly disabled mechanisms: no FAM1 modulation, no FFL, no loss change, no augmentation change, no optimizer change.
- Parameter/runtime/memory impact expected: about half of FAM1+FAM2 `modres`; preflight measured `+8,320` params, roughly `+0.0964%` over ConvIR-B.
- Initialization or no-op behavior: gamma/beta projection weights and bias are zero, so `fam2_modres` must start exactly as original FAM.
- Resume policy: do not resume official original checkpoints into `fam2_modres` for training; first formal comparison is from scratch.
- Defaults changed: none.
- Defaults intentionally preserved: `--fam_mode original`, `--seed -1`, and `--stop_epoch -1`.

## Required Diagnostics

| Diagnostic | Pass line | Artifact |
| --- | --- | --- |
| zero-init equivalence | `max_abs_diff < 1e-6` and `mean_abs_diff < 1e-7` for all three outputs | `fam2_equivalence_*.json` |
| shape/finite train probe | one real Haze4K batch forward/backward has finite content/FFT loss | preflight log |
| gamma/beta stats | every validation epoch logs FAM2 gamma/beta mean/std/min/max and saturation ratios | training log |
| per-sample delta | full Haze4K test CSV for original vs FAM2-only best checkpoints | `fam2_eval_per_image_*.csv` |
| strong-case regression | top 25% original-PSNR group regression count | compare JSON/README |

Gamma/beta interpretation:

- If gamma/beta remain near zero and PSNR is flat, the branch is safe but weak; continue only if full-run cost is cheap and no stronger route is ready.
- If gamma grows quickly and PSNR or color quality drops, constrain the next variant with `gamma = 0.1 * tanh(gamma)` and scaled beta.
- If beta dominates, run a beta-only ablation before FAM1; if gamma dominates, run a gamma-only ablation before FAM1.

## Fair Run Contract

- First hard gate: matched 20 epochs, seed `3407`.
- Command shape: `--num_epoch 1000 --stop_epoch 20 --valid_freq 1 --save_freq 5 --batch_size 8 --learning_rate 4e-4`.
- Candidate command difference: only `--fam_mode fam2_modres`.
- Baseline command difference: only `--fam_mode original`.
- Gamma/beta logging: `--mod_stats_freq 1 --mod_stats_batches 64` for FAM2-only.
- Optimizer and loss: unchanged Adam and original three-scale L1 plus `0.1 * FFT-L1`.
- Full-test comparison: evaluate each run's `Best.pkl`, not only the last epoch.
- Hardware/runtime assumptions: `autodl-dehaze3`, RTX 5090, `convir-cu128`.
- Sample-size policy: full 1000-image Haze4K validation/test for formal gates; subsets only for preflight.

## Gates

| Gate | Continue only if all are true |
| --- | --- |
| preflight | zero-init equivalence passes; finite real-batch backward; parameter delta <= `+0.2%`; default original strict-load remains valid |
| 20 epoch pause rule | if FAM2-only is below matched original by more than `0.10 dB` and most samples have negative delta, pause FAM and inspect implementation/eval equivalence |
| 20 epoch continue rule | if mean PSNR is within `±0.05 dB` and hard/non-homogeneous samples improve with strong-case regression <= `2%`, continue to longer/full FAM2-only |
| 20 epoch repeat rule | if mean PSNR gain is `>= +0.05 dB` and strong-reference regressions are <= `1%`, immediately run a second seed or repeated run |
| FAM1 deferral | do not run FAM1+FAM2 until FAM2-only either passes full/repeat seed or fails in a way that specifically suggests low-resolution modulation is needed |
| FAM1 promotion | run FAM1+FAM2 full only if its 20-epoch gate beats FAM2-only and has no color/texture regression pattern |

## Analysis Plan

- Compare epoch curves for original vs FAM2-only at matched seed and schedule.
- Compare full-test `Best.pkl` PSNR/SSIM, latency, and peak memory.
- Record per-image delta distribution: mean, median, p5/p95, worst-10%, best-10%.
- Bucket by original baseline PSNR: easy top 25%, medium middle 50%, hard bottom 25%.
- Inspect strong-reference regressions first; if easy samples regress systematically, require a gate or amplitude constraint before longer training.
- Keep FAM1+FAM2 and FFL blocked until this card has a written decision.

## Preflight Evidence

- Zero-init equivalence with official `haze4k-base.pkl`: pass; all three outputs have `max_abs_diff = 0.0` and `mean_abs_diff = 0.0`.
- Fresh from-scratch shared initialization check: pass; `602` shared keys checked, `max_abs_diff = 0.0`, no bad shared keys. This prevents the zero-init modulator from silently shifting later baseline-layer initialization.
- Candidate missing keys when loading shared original weights: only `FAM2.modulator.weight` and `FAM2.modulator.bias`.
- Parameter count: original `8,630,665`; FAM2-only `8,638,985`; delta `+8,320` (`+0.0964%`).
- Real Haze4K batch probe: batch size `8`, crop `256`, output shapes `[64, 64]`, `[128, 128]`, `[256, 256]`.
- Real-batch loss after RNG-preserving fix: content `0.6212853`, FFT `11.1377125`, total `1.7350566`; loss finite.
- Gradient health after RNG-preserving fix: finite, grad L2 norm `6.4880692`, grad max abs `0.3279917`.
- Peak CUDA allocated during preflight: `9384.63 MiB`.

## Invalid Attempt

- A first 20-epoch launch was stopped after the original run reached epoch 2.
- Reason: the zero-initialized `nn.Conv2d` modulator consumed RNG during construction before being zeroed, which could shift later shared-layer initialization in from-scratch candidate training.
- Fix: preserve and restore CPU RNG state around modulator construction, then verify fresh shared initialization equality before restarting the gate.
- Remote invalid directory: `results/ConvIR-Haze4K-fam2-modres-stop20-20260531-invalid-rng-init-earlystop`.

## 20-Epoch Gate Evidence

Artifacts:

- Remote run root: `Dehazing/ITS/results/ConvIR-Haze4K-fam2-modres-stop20-20260531`.
- Local logs: `experience_docx/experiment_logs/haze4k_fam2_modres_stop20_20260531/`.
- Original log: `original_train_stop20_seed3407.log`.
- FAM2-only log: `fam2_modres_train_stop20_seed3407.log`.
- Full-test compare JSON: `scout_eval_compare_seed3407_stop20.json`.
- Per-image CSV: `scout_eval_per_image_seed3407_stop20.csv`.
- Bucket analysis JSON: `scout_eval_bucket_analysis_seed3407_stop20.json`.

Matched validation curve:

| Epoch | Original PSNR | FAM2-only PSNR | Delta |
| --- | ---: | ---: | ---: |
| 1 | 20.63 | 20.57 | -0.06 |
| 2 | 20.58 | 21.29 | +0.71 |
| 3 | 21.57 | 21.12 | -0.45 |
| 4 | 22.32 | 21.92 | -0.40 |
| 5 | 22.46 | 22.31 | -0.15 |
| 6 | 22.79 | 22.78 | -0.01 |
| 7 | 22.52 | 22.38 | -0.14 |
| 8 | 22.91 | 23.03 | +0.12 |
| 9 | 22.95 | 23.00 | +0.05 |
| 10 | 22.53 | 22.55 | +0.02 |
| 11 | 23.39 | 23.35 | -0.04 |
| 12 | 23.60 | 23.86 | +0.26 |
| 13 | 24.03 | 23.91 | -0.12 |
| 14 | 23.98 | 24.09 | +0.11 |
| 15 | 24.64 | 23.92 | -0.72 |
| 16 | 23.67 | 24.70 | +1.03 |
| 17 | 24.52 | 24.82 | +0.30 |
| 18 | 24.01 | 24.68 | +0.67 |
| 19 | 24.02 | 24.32 | +0.30 |
| 20 | 22.54 | 22.90 | +0.36 |
| Best | 24.64 | 24.82 | +0.18 |

Full-test `Best.pkl` comparison:

| Metric | Original | FAM2-only | Delta |
| --- | ---: | ---: | ---: |
| PSNR | 24.6424 | 24.8163 | +0.1739 |
| SSIM | 0.947803 | 0.948226 | +0.000424 |
| Avg sync time/img | 0.05378 s | 0.05315 s | -0.00063 s |
| Peak CUDA memory | 570.69 MiB | 571.80 MiB | +1.10 MiB |

Per-image delta:

| Statistic | Value |
| --- | ---: |
| Mean delta | +0.1739 dB |
| Median delta | +0.1408 dB |
| Positive delta ratio | 527/1000 |
| p5 / p95 | -3.1272 / +3.5888 dB |
| Worst 10 mean | -5.4324 dB |
| Best 10 mean | +5.2975 dB |
| All-image regressions <= -0.05 dB | 464/1000 |
| All-image regressions <= -0.20 dB | 444/1000 |
| Strong-reference regressions <= -0.05 dB | 138/250 |

Difficulty buckets by original PSNR:

| Bucket | Original PSNR range | Mean delta | Median delta | Positive ratio | Regressions <= -0.05 dB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hard bottom 25% | 9.71-22.12 | +0.8159 | +1.0359 | 165/250 | 82/250 |
| Medium middle 50% | 22.19-27.25 | +0.0828 | +0.0222 | 253/500 | 244/500 |
| Easy top 25% | 27.28-37.73 | -0.2860 | -0.4084 | 109/250 | 138/250 |

Gamma/beta mechanism stats:

| Epoch | gamma mean | gamma std | gamma abs > 0.5 | beta mean | beta std | beta abs > 0.1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 15 | -0.034241 | 0.210648 | 0.038549 | 0.001411 | 0.085367 | 0.165861 |
| 16 | -0.037317 | 0.229472 | 0.051593 | 0.001302 | 0.092219 | 0.192066 |
| 17 | -0.040835 | 0.234098 | 0.054737 | 0.001762 | 0.096859 | 0.207700 |
| 18 | -0.043832 | 0.238627 | 0.058832 | 0.001601 | 0.097122 | 0.211547 |
| 19 | -0.047427 | 0.233949 | 0.055602 | 0.001236 | 0.099629 | 0.218492 |
| 20 | -0.049957 | 0.223269 | 0.046601 | 0.001280 | 0.103440 | 0.229659 |

Interpretation:

- FAM2-only has a real positive hard-sample signal: hard bottom-25% mean delta is `+0.8159 dB` and median delta is `+1.0359 dB`.
- Preservation fails: easy top-25% mean delta is `-0.2860 dB`, median delta is `-0.4084 dB`, and strong-reference regression is `138/250`, far above the `<= 1%` repeat-seed gate and the `<= 2%` continue gate.
- The branch is active rather than inert. By epoch 20, `|beta| > 0.1` on `22.97%` of sampled activations and `|gamma| > 0.5` on `4.66%`, so the regression pattern is plausibly caused by over-correction rather than no learning.
- FAM2-only is still more promising than the earlier FAM1+FAM2 scout because median delta is positive, the hard bucket improves strongly, and runtime/memory cost is negligible. The current unbounded formulation is not yet safe enough for a longer run.

## Decision

- Decision label: current unbounded FAM2-only fails preservation gate; keep as diagnostic, do not run full or repeat seed yet.
- What this decides next: stay inside the FAM2-only route, but change the mechanism to reduce over-correction before spending on full training.
- Recommended next attempt: FAM2-only constrained modulation with `gamma = 0.1 * tanh(gamma)` and a scaled beta branch, or a cheap gamma-only/beta-only 20-epoch ablation if we want to identify the dominant harmful term first.
- Stop rule for the next attempt: it must keep the hard-sample gain direction while reducing easy top-25% regression count and strong-reference regression below the written gate; otherwise pause FAM before FFL.
