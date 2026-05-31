# Haze4K FAM2 Bounded Modulation

Date: 2026-06-01

Status: gamma-only 20-epoch gate complete; preservation failed, do not run
bounded gamma+relative-beta as the second arm yet.

## Scope

- Project: ConvIR-B dehazing.
- Model family: ConvIR-B.
- Dataset or task: Haze4K image dehazing.
- Branch: `codex/haze4k-fam2-bounded`, based on `codex/haze4k-fam2-only`.
- Main objective: preserve most of the hard-sample gain from FAM2-only while reducing easy/strong-case regression.
- Main metric: PSNR.
- Secondary metrics: SSIM, per-image PSNR delta, hard/easy bucket deltas, strong-reference regression count, gamma/beta activation statistics, latency, peak GPU memory.
- Artifact roots:
  - `experience_docx/experiment_logs/haze4k_fam2_bounded_*`
  - `Dehazing/ITS/results/ConvIR-Haze4K-fam2-bounded-*`

## Baseline Contract

- Official pretrained baseline remains locked: ConvIR-B Haze4K checkpoint reproduced at `34.14 dB PSNR / 0.98971 SSIM`, official table `34.15 / 0.99`.
- Matched from-scratch baseline remains the seed `3407` original 20-epoch run from the FAM2-only card:
  - Original `Best.pkl`: `24.6424 PSNR / 0.947803 SSIM`.
  - Original remote path: `Dehazing/ITS/results/ConvIR-Haze4K-original-stop20-seed3407-20260531/Training-Results/Best.pkl`.
- Unbounded FAM2-only diagnostic result:
  - FAM2 `Best.pkl`: `24.8163 PSNR / 0.948226 SSIM`.
  - Mean delta: `+0.1739 dB`.
  - Hard bottom-25% mean delta: `+0.8159 dB`.
  - Easy top-25% mean delta: `-0.2860 dB`.
  - Strong-reference regressions: `138/250`.
- Reference entrypoint stability: `--fam_mode original` must remain unchanged and strict-load official checkpoints.

## Why This Route

The previous FAM2-only run proved the route is not empty: it improved hard
samples strongly and had positive mean/median delta. It failed because the
same modulation over-corrected easy/strong samples. By epoch 20, `|beta| >
0.1` occurred on `22.97%` of sampled activations, while `|gamma| > 0.5`
occurred on `4.66%`. That makes additive beta the strongest suspect for
easy-case color, brightness, or texture shift.

Therefore the most valuable first question is not "can we get a higher hard
gain?" It is:

```text
Can bounded multiplicative recalibration at FAM2 keep at least half of the
hard-sample gain while removing most easy/strong-case regression?
```

## Variants

### First Formal Gate: Bounded Gamma-Only

- Mode: `fam2_modres_gamma_bounded`.
- Change: FAM2 only; FAM1 remains original.
- Formula:

```python
base = original_FAM(x, cond)
gamma_raw = zero_init_1x1(cond)
gamma = 0.10 * torch.tanh(gamma_raw)
out = base * (1.0 + gamma)
```

- Why first: it directly tests whether the additive beta branch caused most
  easy-case regression. It has the lowest risk because it cannot shift feature
  brightness/color additively.
- Parameter delta expected: `+4,160` params, about `+0.0482%` over ConvIR-B.

### Second Arm If Needed: Bounded Gamma + Relative Beta

- Mode: `fam2_modres_bounded`.
- Change: FAM2 only; FAM1 remains original.
- Formula:

```python
base = original_FAM(x, cond)
gamma_raw, beta_raw = zero_init_1x1(cond).chunk(2, dim=1)
gamma = 0.10 * torch.tanh(gamma_raw)
scale = base.detach().std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
beta = 0.05 * scale * torch.tanh(beta_raw)
out = base * (1.0 + gamma) + beta
```

- Why second: if gamma-only fixes easy regression but loses too much hard gain,
  a small relative beta can restore local correction while staying tied to the
  current feature scale.
- Parameter delta expected: `+8,320` params, about `+0.0964%` over ConvIR-B.

## Explicitly Not Running Yet

- No FAM1.
- No FAM1+FAM2.
- No FFL.
- No loss change.
- No optimizer, augmentation, crop, or schedule change.
- No full/repeat seed before this route passes the 20-epoch preservation gate.

## Preflight

Run for each mode before training:

```bash
cd /root/autodl-tmp/workspace/ConvIR-B/Dehazing/ITS
/root/miniconda3/envs/convir-cu128/bin/python \
  /root/autodl-tmp/workspace/ConvIR-B/experience_docx/tools/check_haze4k_fam_equivalence.py \
  --candidate_mode fam2_modres_gamma_bounded \
  --seed 3407 \
  --output /root/autodl-tmp/workspace/ConvIR-B/experience_docx/experiment_logs/haze4k_fam2_bounded_preflight_20260601/fam2_gamma_bounded_equivalence_seed3407.json

/root/miniconda3/envs/convir-cu128/bin/python \
  /root/autodl-tmp/workspace/ConvIR-B/experience_docx/tools/preflight_haze4k_fam2.py \
  --data_dir /root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K \
  --mode fam2_modres_gamma_bounded \
  --seed 3407 \
  --output /root/autodl-tmp/workspace/ConvIR-B/experience_docx/experiment_logs/haze4k_fam2_bounded_preflight_20260601/fam2_gamma_bounded_real_batch_seed3407.json
```

Pass line:

- zero-init output equivalence: `max_abs_diff < 1e-6`, `mean_abs_diff < 1e-7`;
- fresh shared initialization: no shared-key drift;
- finite real-batch content/FFT loss and finite gradients;
- default `original` entrypoint unchanged;
- parameter delta within `+0.2%`.

## Required Logs

Every formal 20-epoch bounded run must produce these text artifacts:

| Required artifact | Tool or source |
| --- | --- |
| per-sample PSNR/SSIM delta for Best | `eval_haze4k_checkpoint_compare.py`, tag `*_best` |
| per-sample PSNR/SSIM delta for Last | `eval_haze4k_checkpoint_compare.py`, tag `*_last` |
| direct Best-vs-Last checkpoint comparison | `eval_haze4k_checkpoint_compare.py` with `--original_mode` set to the candidate mode |
| easy/medium/hard bucket delta for Best and Last | `analyze_haze4k_delta_buckets.py` |
| strong-reference regression count for Best and Last | compare JSON plus bucket analysis JSON |
| gamma mean/std/min/max and beta mean/std/min/max | train `MOD_STATS` plus modulation bucket JSON |
| `|gamma| > 0.05 / 0.10` proportions | train `MOD_STATS` plus modulation bucket JSON |
| `|beta| > 0.02 / 0.05` proportions | train `MOD_STATS` plus modulation bucket JSON |
| easy/medium/hard gamma/beta distribution | `analyze_haze4k_modulation_buckets.py` |

The modulation bucket analysis is a hard requirement. A bounded FAM2 run is only
healthy if the hard bucket receives larger modulation than the easy bucket. If
easy samples are still strongly modulated, the next mechanism should be a
haze-aware gate rather than a larger beta branch.

## 20-Epoch Gate

First run only `fam2_modres_gamma_bounded` unless preflight fails.

Command shape:

```bash
python main.py \
  --mode train \
  --model_name ConvIR-Haze4K-fam2_modres_gamma_bounded-stop20-seed3407-20260601 \
  --data Haze4K \
  --data_dir /root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K \
  --version base \
  --fam_mode fam2_modres_gamma_bounded \
  --batch_size 8 \
  --learning_rate 4e-4 \
  --num_epoch 1000 \
  --stop_epoch 20 \
  --print_freq 50 \
  --num_worker 8 \
  --save_freq 5 \
  --valid_freq 1 \
  --seed 3407 \
  --mod_stats_freq 1 \
  --mod_stats_batches 64
```

Compare `Best.pkl` against the locked original 20-epoch `Best.pkl` using:

- `experience_docx/tools/eval_haze4k_checkpoint_compare.py`
- `experience_docx/tools/analyze_haze4k_delta_buckets.py`
- `experience_docx/tools/analyze_haze4k_modulation_buckets.py`

The run script at
`experience_docx/experiment_logs/haze4k_fam2_bounded_gamma_stop20_20260601/run_gamma_only_stop20.sh`
evaluates both `Best.pkl` and `Final.pkl`, then performs a direct Best-vs-Last
checkpoint comparison.

## Gates

| Gate | Continue only if true |
| --- | --- |
| no collapse | candidate Best PSNR is not below original Best by more than `0.10 dB` |
| hard retention | hard bottom-25% mean delta is at least `+0.40 dB`, or hard median delta is at least `+0.50 dB` |
| easy preservation | easy top-25% mean delta is no worse than `-0.05 dB` |
| regression cap | strong-reference regressions <= `25/250` for another bounded run; <= `5/250` for repeat/full promotion |
| mechanism sanity | bounded gamma does not sit saturated at `abs(gamma) > 0.09` for most sampled activations |
| cost | params <= ConvIR-B `+0.2%`, latency <= `+10%`, peak memory fits current card |

Decision rules:

- If gamma-only passes hard retention and easy preservation, promote gamma-only
  to a second seed or longer run before testing beta again.
- If gamma-only fixes easy preservation but hard retention is too weak, run
  `fam2_modres_bounded` as the second arm.
- If gamma-only still has easy/strong-case regression, do not add beta; inspect
  whether FAM2 conditioning needs an input-dependent gate or whether the FAM
  route should pause before FFL.
- If gamma-only collapses by more than `0.10 dB`, first inspect equivalence,
  train/eval pipeline, and shared initialization before any new mechanism.

## Gamma-Only Gate Evidence

Artifacts:

- Remote run root: `Dehazing/ITS/results/ConvIR-Haze4K-fam2-bounded-gamma-stop20-20260601`.
- Local logs: `experience_docx/experiment_logs/haze4k_fam2_bounded_gamma_stop20_20260601/`.
- Training log: `fam2_modres_gamma_bounded_train_stop20_seed3407.log`.
- Best comparison: `scout_eval_compare_seed3407_stop20_best.json`.
- Last comparison: `scout_eval_compare_seed3407_stop20_last.json`.
- Best-vs-Last comparison: `scout_eval_compare_seed3407_stop20_best_vs_last.json`.
- Best modulation bucket analysis: `modulation_bucket_analysis_seed3407_stop20_best.json`.
- Last modulation bucket analysis: `modulation_bucket_analysis_seed3407_stop20_last.json`.

Preflight:

- Zero-init equivalence: pass; all three outputs have `max_abs_diff = 0.0` and `mean_abs_diff = 0.0`.
- Fresh shared initialization: pass; `602` shared keys checked, `max_abs_diff = 0.0`.
- Parameter count: original `8,630,665`; gamma-only `8,634,825`; delta `+4,160` (`+0.0482%`).
- Real-batch probe: finite content loss `0.6212853`, FFT loss `11.1377125`, total `1.7350566`, grad L2 `6.4825544`, peak CUDA memory `9416.61 MiB`.

Matched validation curve:

| Epoch | Gamma-only PSNR |
| --- | ---: |
| 1 | 20.52 |
| 2 | 20.99 |
| 3 | 20.53 |
| 4 | 22.22 |
| 5 | 22.09 |
| 6 | 22.63 |
| 7 | 22.30 |
| 8 | 23.56 |
| 9 | 22.65 |
| 10 | 22.21 |
| 11 | 23.26 |
| 12 | 23.76 |
| 13 | 23.71 |
| 14 | 22.86 |
| 15 | 23.88 |
| 16 | 23.39 |
| 17 | 24.21 |
| 18 | 24.62 |
| 19 | 24.19 |
| 20 | 23.74 |
| Best | 24.62 |

Best checkpoint full-test comparison against matched original:

| Metric | Original Best | Gamma-only Best | Delta |
| --- | ---: | ---: | ---: |
| PSNR | 24.6424 | 24.6153 | -0.0271 |
| SSIM | 0.947803 | 0.939256 | -0.008546 |
| Median PSNR delta | n/a | n/a | +0.2570 |
| Strong-reference regressions <= -0.05 dB | n/a | n/a | 181/250 |
| All-image regressions <= -0.20 dB | n/a | n/a | 424/1000 |

Best checkpoint difficulty buckets:

| Bucket | Mean delta | Median delta | Regressions <= -0.05 dB | gamma abs mean | `|gamma| > 0.05` | `|gamma| > 0.09` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hard bottom 25% | +0.8054 | +0.8222 | 61/250 | 0.026507 | 0.149840 | 0.006122 |
| Medium middle 50% | +0.1800 | +0.4118 | 211/500 | 0.027359 | 0.163024 | 0.006693 |
| Easy top 25% | -1.2740 | -1.1545 | 181/250 | 0.028362 | 0.173574 | 0.008790 |

Last checkpoint:

| Metric | Value |
| --- | ---: |
| Mean PSNR delta vs original Best | -0.9060 |
| Median PSNR delta vs original Best | -0.5858 |
| Hard bottom-25% mean delta | +1.1242 |
| Easy top-25% mean delta | -3.0049 |
| Strong-reference regressions <= -0.05 dB | 206/250 |
| Best-vs-Last mean PSNR delta | -0.8789 |

Mechanism interpretation:

- Gamma-only preserves hard-sample gain direction: hard bottom-25% mean delta is `+0.8054 dB`.
- It fails the easy-preservation and regression gates badly: easy top-25% mean delta is `-1.2740 dB`, and strong-reference regressions are `181/250`.
- The modulation bucket check is decisive: easy samples are modulated more than hard samples (`gamma_abs_mean 0.028362` vs `0.026507`, `|gamma| > 0.05` ratio `0.173574` vs `0.149840`). This is the opposite of the desired behavior.
- Since gamma-only also over-corrects easy samples, additive beta was not the sole cause of the previous regression pattern.
- Last checkpoint is much worse than Best and should not be used for any promotion decision.

## Decision

- Decision label: bounded gamma-only fails preservation; keep as diagnostic and do not promote.
- Do not run `fam2_modres_bounded` as the second arm under the current plan. Gamma-only did not fix easy/strong-case regression, so adding relative beta is unlikely to answer the highest-value question.
- Next most valuable mechanism: FAM2 modulation needs a haze-aware or confidence gate that suppresses modulation on already-clean/easy samples, rather than merely bounding gamma/beta amplitude.
- FAM route status: hard-sample signal remains real, so do not discard FAM2; however, no FAM2 variant should enter full/repeat seed until easy bucket modulation becomes lower than hard bucket modulation and strong-reference regressions fall below the written gate.
