# Haze4K FAM2 Bounded Modulation

Date: 2026-06-01

Status: design ready; preflight required before any training run.

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

## Decision

- Decision label: start with bounded gamma-only as the most informative next
  gate.
- Rationale: previous evidence points to additive beta as the higher-risk term;
  gamma-only is the cleanest way to test whether FAM2 can become a safe
  architecture change rather than a hard-sample-only diagnostic.
- Next action after this card: sync code to `autodl-dehaze3`, run preflight for
  `fam2_modres_gamma_bounded`, then run the 20-epoch gate only if preflight
  passes.
