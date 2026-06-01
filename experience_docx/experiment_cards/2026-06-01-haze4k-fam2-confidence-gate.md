# Haze4K FAM2 Confidence-Gated Gamma-Only

Date: 2026-06-01

Status: 20-epoch gate complete; Best quality is positive, but preservation and
gate-budget behavior fail promotion.

## Scope

- Project: ConvIR-B dehazing.
- Dataset or task: Haze4K image dehazing.
- Branch: `codex/haze4k-fam2-confidence-gate`, based on `codex/haze4k-fam2-bounded`.
- First formal arm: `fam2_modres_gamma_conf_gated`.
- Main objective: keep the FAM2 hard-sample gain while suppressing modulation on easy or high-confidence samples.
- Frozen variables: no beta, no FAM1 modulation, no FFL, no optimizer change, no crop change, no schedule change.
- Artifact roots:
  - `experience_docx/experiment_logs/haze4k_fam2_conf_gate_stop20_20260601/`
  - `Dehazing/ITS/results/ConvIR-Haze4K-fam2-conf-gamma-stop20-20260601/`

## Baseline Evidence

Matched 20-epoch original baseline:

- Original `Best.pkl`: `24.6424 PSNR / 0.947803 SSIM`.
- Original remote path: `Dehazing/ITS/results/ConvIR-Haze4K-original-stop20-seed3407-20260531/Training-Results/Best.pkl`.

Unbounded FAM2-only diagnostic:

- Mean PSNR delta: `+0.1739 dB`.
- Hard bottom-25% mean delta: `+0.8159 dB`.
- Easy top-25% mean delta: `-0.2860 dB`.
- Strong-reference regressions: `138/250`.

Bounded gamma-only diagnostic:

- Mean PSNR delta: `-0.0271 dB`.
- Median PSNR delta: `+0.2570 dB`.
- Hard bottom-25% mean delta: `+0.8054 dB`.
- Easy top-25% mean delta: `-1.2740 dB`.
- Strong-reference regressions: `181/250`.
- Mechanism failure: easy samples had larger gamma activity than hard samples (`gamma_abs_mean 0.028362` vs `0.026507`; `|gamma| > 0.05` ratio `0.173574` vs `0.149840`).
- Last checkpoint was not promotable: Best-vs-Last mean PSNR delta `-0.8789 dB`.

Decision carried forward: FAM2 hard-sample signal is real, but amplitude bounds alone do not create correct selectivity. The next variable must be sample-level confidence gating.

## Hypothesis

```text
If FAM2 bounded gamma is multiplied by an image-level confidence gate, and
training adds a small budget pressure against opening the gate on low-loss
samples, then hard bottom-25% samples should keep positive FAM2 gain while easy
top-25% and strong-reference regressions fall sharply.
```

## Change

- Mode: `fam2_modres_gamma_conf_gated`.
- FAM2 only; FAM1 remains original.
- Gamma is still zero-initialized and bounded.
- Gate is image-level scalar, not spatial.
- Gate starts at `0.5`; it is not initialized near zero.
- Gate descriptor uses detached global mean/std statistics from SCM condition features and detached fused FAM features, so the gate learns selection without reshaping upstream features through the gate penalty.

Formula:

```python
fused = original_FAM_merge(x1, cond)
gamma_raw = zero_init_1x1_gamma(cond)
gamma_base = 0.10 * torch.tanh(gamma_raw)

gate = sigmoid(linear(global_stats(cond.detach(), fused.detach())))
gamma = gate.view(B, 1, 1, 1) * gamma_base

out = fused * (1.0 + gamma)
```

Gate budget loss:

```python
restore_loss_per_image = per_image_restore_loss.detach()
easy_weight = 1.0 - rank(restore_loss_per_image) / (batch_size - 1)
L_total = L_restore + lambda_gate * mean(easy_weight * gate)
```

Default first run:

- `lambda_gate = 0.02`
- `gate_warmup_epochs = 5`
- `gate_ramp_epochs = 5`

Rationale: gamma starts at zero, so early gate gradients are weak. The first 5 epochs let gamma leave zero before the gate budget becomes active.

## Proxy Preflight

Before spending the 20-epoch run, compute deployable proxy separability using the existing per-image CSV and test inputs.

Required tool:

```bash
python /root/autodl-tmp/workspace/ConvIR-B/experience_docx/tools/analyze_haze4k_proxy_separability.py \
  --data_dir /root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K \
  --compare_csv /root/autodl-tmp/workspace/ConvIR-B/Dehazing/ITS/results/ConvIR-Haze4K-fam2-bounded-gamma-stop20-20260601/logs/scout_eval_per_image_seed3407_stop20_best.csv \
  --baseline_checkpoint /root/autodl-tmp/workspace/ConvIR-B/Dehazing/ITS/results/ConvIR-Haze4K-original-stop20-seed3407-20260531/Training-Results/Best.pkl \
  --output_json /root/autodl-tmp/workspace/ConvIR-B/Dehazing/ITS/results/ConvIR-Haze4K-fam2-conf-gamma-stop20-20260601/logs/proxy_separability_seed3407.json \
  --output_csv /root/autodl-tmp/workspace/ConvIR-B/Dehazing/ITS/results/ConvIR-Haze4K-fam2-conf-gamma-stop20-20260601/logs/proxy_separability_seed3407.csv
```

Proxy preflight is supportive, not a replacement for the training gate. Continue only if at least one deployable proxy has stable hard/easy direction (`best_direction_auc >= 0.60`) or the proxy report gives a clear reason to proceed despite weak separability.

## Required Logs

| Required artifact | Tool or source |
| --- | --- |
| zero-init equivalence | `check_haze4k_fam_equivalence.py` |
| one real-batch finite backward | `preflight_haze4k_fam2.py` |
| proxy separability JSON/CSV | `analyze_haze4k_proxy_separability.py` |
| train log with gate/gamma stats | `main.py --mod_stats_freq 1` |
| Best per-image PSNR/SSIM delta | `eval_haze4k_checkpoint_compare.py` |
| Last per-image PSNR/SSIM delta | `eval_haze4k_checkpoint_compare.py` |
| Best-vs-Last direct comparison | `eval_haze4k_checkpoint_compare.py` |
| hard/easy delta buckets | `analyze_haze4k_delta_buckets.py` |
| gate/effective-gamma buckets | `analyze_haze4k_modulation_buckets.py` |

## 20-Epoch Gate

Command shape:

```bash
python main.py \
  --mode train \
  --model_name ConvIR-Haze4K-fam2_modres_gamma_conf_gated-stop20-seed3407-20260601 \
  --data Haze4K \
  --data_dir /root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K \
  --version base \
  --fam_mode fam2_modres_gamma_conf_gated \
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
  --mod_stats_batches 64 \
  --gate_lambda 0.02 \
  --gate_warmup_epochs 5 \
  --gate_ramp_epochs 5
```

| Gate | Continue only if true |
| --- | --- |
| no collapse | candidate Best PSNR is not below original Best by more than `0.10 dB` |
| SSIM safety | Best SSIM delta is no worse than `-0.002` unless PSNR/gate evidence strongly justifies another scout |
| hard retention | hard bottom-25% mean delta is at least `+0.40 dB`, or hard median delta is at least `+0.50 dB` |
| easy preservation | easy top-25% mean delta is no worse than `-0.05 dB` |
| regression cap | strong-reference regressions <= `25/250` for another gated run; <= `5/250` for repeat/full promotion |
| gate selectivity | `gate_mean_hard > gate_mean_easy` |
| effective modulation | `effective_gamma_abs_mean_hard > effective_gamma_abs_mean_easy` |
| Best-vs-Last | record Last; do not promote if Last shows catastrophic degradation |
| cost | params <= ConvIR-B `+0.2%`, latency <= `+10%`, peak memory fits current card |

## Decision Rules

- If gated gamma-only passes hard retention, easy preservation, regression cap, and gate/effective-gamma selectivity, repeat the 20-epoch run or run a second seed before adding beta.
- If easy preservation passes but hard gain drops too much, run gated bounded gamma+relative-beta with the same gate applied to both terms.
- If easy regression remains severe, or if gate/effective gamma is still easy >= hard, pause the FAM route. Do not add FAM1, spatial gate, large beta, or FFL before the selectivity failure is explained.

## 20-Epoch Gate Evidence

Artifacts:

- Run root: `Dehazing/ITS/results/ConvIR-Haze4K-fam2-conf-gamma-stop20-20260601`.
- Training checkpoint root: `Dehazing/ITS/results/ConvIR-Haze4K-fam2_modres_gamma_conf_gated-stop20-seed3407-20260601/Training-Results/`.
- Local log copy: `experience_docx/experiment_logs/haze4k_fam2_conf_gate_stop20_20260601/`.
- Best comparison: `scout_eval_compare_seed3407_stop20_best.json`.
- Best bucket analysis: `scout_eval_bucket_analysis_seed3407_stop20_best.json`.
- Best modulation bucket analysis: `modulation_bucket_analysis_seed3407_stop20_best.json`.
- Last comparison: `scout_eval_compare_seed3407_stop20_last.json`.
- Best-vs-Last comparison: `scout_eval_compare_seed3407_stop20_best_vs_last.json`.

Preflight:

- Zero-init equivalence: pass; all output `max_abs_diff = 0.0` and `mean_abs_diff = 0.0`.
- Fresh shared initialization: pass; `602` shared keys checked, `max_abs_diff = 0.0`.
- Parameter count: original `8,630,665`; gated gamma `8,635,082`; delta `+4,417` (`+0.0512%`).
- Real-batch probe: finite content loss `0.6212853`, FFT loss `11.1377125`, total `1.7350566`, grad L2 `6.4825469`, peak CUDA memory `9456.87 MiB`.

Proxy separability:

| Proxy | Direction | Hard mean | Easy mean | Hard/easy AUC |
| --- | --- | ---: | ---: | ---: |
| baseline residual abs p95 | higher on hard | 0.305467 | 0.195128 | 0.797136 |
| baseline residual abs mean | higher on hard | 0.080173 | 0.048841 | 0.786304 |
| input saturation mean | lower on hard | 0.111473 | 0.201808 | 0.760992 best-direction |
| input brightness mean | higher on hard | 0.610981 | 0.505571 | 0.756912 |

This passes the proxy preflight: deployable image/baseline-output statistics do
separate hard and easy buckets.

Matched validation curve:

| Epoch | PSNR | Gate mean | Effective gamma abs mean |
| --- | ---: | ---: | ---: |
| 1 | 20.62 | 0.658318 | 0.00492019 |
| 2 | 20.28 | 0.837691 | 0.01126576 |
| 3 | 18.71 | 0.923589 | 0.01546411 |
| 4 | 21.74 | 0.962248 | 0.01904924 |
| 5 | 22.41 | 0.976829 | 0.01947085 |
| 6 | 23.17 | 0.120719 | 0.00254801 |
| 7 | 22.80 | 0.012100 | 0.00024385 |
| 8 | 23.11 | 0.004357 | 0.00008475 |
| 9 | 22.83 | 0.002335 | 0.00004447 |
| 10 | 23.48 | 0.001242 | 0.00002336 |
| 11 | 23.18 | 0.000750 | 0.00001404 |
| 12 | 24.23 | 0.000479 | 0.00000829 |
| 13 | 23.53 | 0.000287 | 0.00000524 |
| 14 | 23.89 | 0.000375 | 0.00000636 |
| 15 | 24.94 | 0.000108 | 0.00000172 |
| 16 | 24.54 | 0.000235 | 0.00000385 |
| 17 | 25.09 | 0.000192 | 0.00000331 |
| 18 | 24.53 | 0.000085 | 0.00000148 |
| 19 | 24.03 | 0.000108 | 0.00000179 |
| 20 | 24.54 | 0.000103 | 0.00000159 |
| Best | 25.09 | n/a | n/a |

The budget behavior is decisive: during warmup the gate opened almost fully,
then the rank-weighted budget closed it almost completely by epoch 7. This
means the run is not evidence for a healthy active gated modulator; it is
evidence that the first budget strength is too aggressive.

Best checkpoint full-test comparison against matched original:

| Metric | Original Best | Gated gamma Best | Delta |
| --- | ---: | ---: | ---: |
| PSNR | 24.6424 | 25.0948 | +0.4523 |
| SSIM | 0.947803 | 0.952105 | +0.004303 |
| Median PSNR delta | n/a | n/a | +0.4614 |
| Strong-reference regressions <= -0.05 dB | n/a | n/a | 121/250 |
| All-image regressions <= -0.20 dB | n/a | n/a | 350/1000 |
| Peak CUDA memory | 570.69 MiB | 579.68 MiB | +8.98 MiB |

Best checkpoint difficulty buckets:

| Bucket | Mean delta | Median delta | Regressions <= -0.05 dB | Gate mean | Effective gamma abs mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hard bottom 25% | +0.9380 | +0.8725 | 75/250 | 0.000157 | 0.00000282 |
| Medium middle 50% | +0.4707 | +0.5184 | 190/500 | 0.000079 | 0.00000143 |
| Easy top 25% | -0.0700 | -0.0046 | 121/250 | 0.000035 | 0.00000069 |

Last checkpoint:

| Metric | Value |
| --- | ---: |
| Mean PSNR delta vs original Best | -0.1036 |
| Median PSNR delta vs original Best | -0.0557 |
| Hard bottom-25% mean delta | +1.5366 |
| Easy top-25% mean delta | -1.8973 |
| Strong-reference regressions <= -0.05 dB | 195/250 |
| Best-vs-Last mean PSNR delta | -0.5559 |
| Best-vs-Last mean SSIM delta | -0.010278 |

## Decision

- Decision label: `fam2_modres_gamma_conf_gated` is positive for Best quality
  and hard-sample gain, but fails preservation and repeat/full promotion.
- Passing gates: no collapse, SSIM safety, hard retention, cost, and direction
  of gate/effective-gamma selectivity (`hard > easy`).
- Failed gates: easy preservation (`-0.0700 dB`, target `>= -0.05`),
  strong-reference regression cap (`121/250`, target `<= 25/250` for another
  gated run), and Best-vs-Last stability (`-0.5559 dB`).
- Mechanism interpretation: the gate did rank hard above easy, but the absolute
  gate and effective gamma were near zero. The Best gain is therefore not
  clean evidence that active FAM2 modulation solved the route; it is more likely
  a training-trajectory effect after the gate was aggressively suppressed.
- Do not run a repeat seed or full training for this exact budget.
- Do not add beta yet. The next useful run should stay gamma-only and change
  only the gate budget, for example much smaller `lambda_gate`, shorter ramp
  shock, or a target-budget/hinge penalty that discourages easy gates without
  globally driving all gates to zero.
