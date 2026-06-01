# Haze4K Hard-Aware Frequency Loss

Date: 2026-06-01

Status: authorized for cloud stop20 scout.

## Scope

- Project: ConvIR-B dehazing.
- Model family: ConvIR-B.
- Dataset or task: Haze4K image dehazing.
- Branch: `codex/haze4k-hardfreq-loss`.
- Primary objective: test whether a conservative hard-sample FFT boost can
  improve weak Haze4K cases after the FAM2 gate route failed deployable
  selectivity.
- Main metric: PSNR.
- Secondary metrics: SSIM, hard/medium/easy PSNR deltas, strong-reference
  regression count, Best-vs-Last stability, training loss health, latency, peak
  GPU memory.
- Execution environment: cloud only, currently `autodl-dehaze3`.
- Artifact roots:
  - `experience_docx/experiment_logs/haze4k_hardfreq_loss_stop20_20260601/`
  - `Dehazing/ITS/results/ConvIR-Haze4K-hardfreq-loss-stop20-20260601/`

## Baseline Contract

- Official pretrained baseline remains locked: ConvIR-B Haze4K checkpoint
  reproduced at `34.14 dB PSNR / 0.98971 SSIM`, official table `34.15 / 0.99`.
- Matched from-scratch baseline remains the seed `3407` original 20-epoch run:
  `24.6424 PSNR / 0.947803 SSIM`.
- Original baseline checkpoint:
  `Dehazing/ITS/results/ConvIR-Haze4K-original-stop20-seed3407-20260531/Training-Results/Best.pkl`.
- Stop20 original seed noise floor is high: mean PSNR sample std `0.2206 dB`
  across seeds `3407/2027/8675`, with hard bucket std `0.4551 dB`.
- Reference entrypoint stability: `--fam_mode original --loss_mode original`
  must remain the original ConvIR-B training path.

## Why This Is Next

The FAM2 modulation sequence found a real hard-sample signal but failed the
deployable selectivity gate:

- FAM2-only hard bottom-25% mean delta was `+0.8159 dB`, but easy top-25% was
  `-0.2860 dB`.
- Confidence-gated gamma Best reached `+0.4523 dB`, but the active gate collapsed
  nearly to zero and strong-reference regressions remained `121/250`.
- Offline selector meta-analysis ended with `FAIL_STOP_FAM_ROUTE`; no deployable
  proxy reached the required AUC and threshold-gate floor.

Therefore the next useful route should not spend another run on FAM gates or a
larger architecture. It should keep the inference graph unchanged and ask
whether the current training objective under-serves hard/frequency recovery.

## Hypothesis

```text
If batch-hard samples receive a small extra frequency-domain loss, hard
bottom-25% Haze4K quality should improve without introducing FAM-style
inference-time regressions, because the intervention changes only the training
emphasis and leaves ConvIR-B architecture unchanged.
```

## Change

- `models/ConvIR.py` remains unchanged.
- `--fam_mode original` remains the only model mode for this route.
- New training option: `--loss_mode hard_fft_boost`.
- New scalar: `--hard_fft_lambda`, first value `0.02`.
- Base loss remains `L_content + 0.1 * L_fft`.
- Extra loss:

```python
restore_loss_per_image = (loss_content_per_image + 0.1 * loss_fft_per_image).detach()
hard_weight = rank_from_easy_0_to_hard_1(restore_loss_per_image)
hard_fft_loss = mean(hard_weight * loss_fft_per_image)
loss = loss_content + 0.1 * loss_fft + hard_fft_lambda * hard_fft_loss
```

The hard ranking is batch-local and detached. It does not use test labels,
teacher outputs, deployable proxies, or any inference-time branch.

## Explicitly Not Running Yet

- No FAM2 retry, gate penalty, beta branch, FAM1, SCM, PFFB, RHFD, or PFD-ConvIR.
- No teacher preservation.
- No FFL matrix weighting.
- No optimizer, crop, augmentation, architecture, or checkpoint-loading change.
- No promotion claim from one stop20 seed because the measured stop20 noise
  floor is too high.

## Preflight

Run on cloud before training:

```bash
cd /root/autodl-tmp/workspace/ConvIR-B-hardfreq-loss/Dehazing/ITS
/root/miniconda3/envs/convir-cu128/bin/python \
  /root/autodl-tmp/workspace/ConvIR-B-hardfreq-loss/experience_docx/tools/preflight_haze4k_hardfreq_loss.py \
  --data_dir /root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K \
  --seed 3407 \
  --batch_size 8 \
  --hard_fft_lambda 0.02 \
  --output /root/autodl-tmp/workspace/ConvIR-B-hardfreq-loss/Dehazing/ITS/results/ConvIR-Haze4K-hardfreq-loss-stop20-20260601/logs/hardfreq_loss_preflight_seed3407.json
```

Pass line:

- output shapes match the original multi-scale contract;
- hard weights cover `[0, 1]` for batch size 8;
- content, FFT, hard FFT, and total losses are finite;
- gradients are finite;
- peak memory fits current GPU.

## 20-Epoch Scout

Command shape:

```bash
python main.py \
  --mode train \
  --model_name ConvIR-Haze4K-hardfreq-loss-stop20-seed3407-20260601 \
  --data Haze4K \
  --data_dir /root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K \
  --version base \
  --fam_mode original \
  --loss_mode hard_fft_boost \
  --hard_fft_lambda 0.02 \
  --batch_size 8 \
  --learning_rate 4e-4 \
  --num_epoch 1000 \
  --stop_epoch 20 \
  --print_freq 50 \
  --num_worker 8 \
  --save_freq 5 \
  --valid_freq 1 \
  --seed 3407
```

## Required Logs

| Required artifact | Tool or source |
| --- | --- |
| finite loss/grad preflight | `preflight_haze4k_hardfreq_loss.py` |
| train log with hard FFT loss | `main.py --loss_mode hard_fft_boost` |
| Best per-image PSNR/SSIM delta | `eval_haze4k_checkpoint_compare.py` |
| Last per-image PSNR/SSIM delta | `eval_haze4k_checkpoint_compare.py` |
| Best-vs-Last direct comparison | `eval_haze4k_checkpoint_compare.py` |
| hard/easy delta buckets | `analyze_haze4k_delta_buckets.py` |

## Gates

| Gate | Continue only if true |
| --- | --- |
| no collapse | candidate Best PSNR is not below original Best by more than `0.10 dB` |
| hard movement | hard bottom-25% mean delta is positive, or hard median delta is positive with no global collapse |
| easy preservation | easy top-25% mean delta is no worse than `-0.05 dB` |
| regression cap | strong-reference regressions <= `25/250` for another stop20 run |
| stability | Best-vs-Last mean PSNR delta is not worse than `-0.30 dB` |
| loss health | hard FFT loss is finite and does not dominate the restore loss curve |
| cost | inference params/FLOPs unchanged; peak memory fits current card |

## Decision Rules

- If quality, hard movement, easy preservation, and stability all pass, repeat
  stop20 on at least one additional seed before promotion.
- If hard movement passes but easy preservation fails mildly, try a smaller
  `hard_fft_lambda` such as `0.01`; do not move to teacher yet.
- If hard movement fails and easy preservation is neutral, close this exact
  hard FFT boost and consider a true FFL preflight.
- If it collapses, restore the original loss path and inspect schedule/noise
  before any new mechanism.
