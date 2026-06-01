# Haze4K Hard-Aware Frequency Loss

Date: 2026-06-01

Status: completed diagnostic-only; exact `hard_fft_lambda=0.02` route failed
the stop20 gate and should not be promoted or repeated as-is.

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

## Stop20 Outcome

Cloud run:

- Host: `autodl-dehaze3`.
- Local evidence package:
  `experience_docx/experiment_logs/haze4k_hardfreq_loss_stop20_20260601/`.
- Cloud raw root:
  `Dehazing/ITS/results/ConvIR-Haze4K-hardfreq-loss-stop20-20260601/`.
- Candidate checkpoint family:
  `ConvIR-Haze4K-hardfreq-loss-stop20-seed3407-20260601`.
- Baseline comparator:
  `ConvIR-Haze4K-original-stop20-seed3407-20260531/Training-Results/Best.pkl`.

Preflight passed the mechanical safety checks:

- output shapes matched the original multi-scale contract;
- hard weights covered `[0.0, 1.0]` for batch size 8;
- content, FFT, hard FFT, total loss, and gradients were finite;
- peak CUDA memory was `9288.85 MiB`.

Best checkpoint comparison against the matched original stop20 baseline:

| Metric | Original Best | HardFreq Best | Delta |
| --- | ---: | ---: | ---: |
| Mean PSNR | `24.6424` | `24.4298` | `-0.2127` |
| Mean SSIM | `0.947803` | `0.942971` | `-0.004832` |

Best bucket behavior:

| Bucket by original PSNR | Mean PSNR delta | Median PSNR delta | Positive count |
| --- | ---: | ---: | ---: |
| hard bottom 25% | `+0.5999` | `+0.3879` | `154/250` |
| medium middle 50% | `-0.1072` | `-0.1670` | `232/500` |
| easy top 25% | `-1.2363` | `-1.3037` | `57/250` |

Regression and stability evidence:

- strong-reference regressions at `delta <= -0.05`: `188/250`;
- all-image regressions at `delta <= -0.20`: `506/1000`;
- Last checkpoint: `23.7376 PSNR / 0.928821 SSIM`;
- Best-vs-Last mean PSNR delta: `-0.6922 dB`.

## Gate Verdict

| Gate | Verdict | Evidence |
| --- | --- | --- |
| no collapse | FAIL | Best mean PSNR delta `-0.2127 dB`, below the `-0.10 dB` floor |
| hard movement | PASS | hard bottom 25% mean delta `+0.5999 dB` |
| easy preservation | FAIL | easy top 25% mean delta `-1.2363 dB`, below the `-0.05 dB` floor |
| regression cap | FAIL | strong-reference regressions `188/250`, above cap `25/250` |
| stability | FAIL | Best-vs-Last mean delta `-0.6922 dB`, below the `-0.30 dB` floor |
| loss health | PASS | preflight, train log, and epoch summaries stayed finite |
| cost | PASS | inference graph unchanged; memory fit the card |

Decision label: `FAIL_STOP_HARDFFT_LAMBDA_002`.

Interpretation: the hard-frequency emphasis finds a real hard-sample direction,
but it buys that gain by damaging easy cases, increasing strong-reference
regressions, and producing poor Best-to-Last stability. This exact loss setting
is diagnostic-only. Do not promote it, repeat it as-is, or spend a long run on
`hard_fft_lambda=0.02`.

Useful follow-up boundary:

- Keep the result as evidence that hard/frequency weighting can move the
  hard-bottom bucket.
- Any future frequency route must first reduce global/easy collateral damage,
  for example through a smaller weight, a safer schedule, or a different
  frequency weighting formulation; it still needs a fresh preflight and gate
  card before training.

## Synced Text Artifacts

The GitHub-facing evidence for this stop20 scout is intentionally text-only:

- `hardfreq_loss_preflight_seed3407.json`
- `hardfreq_loss_preflight_seed3407.manual.json`
- `hardfreq_loss_train_stop20_seed3407.log`
- `status.txt`
- `run_hardfreq_loss_stop20.sh`
- `scout_eval_compare_seed3407_stop20_best.json`
- `scout_eval_compare_seed3407_stop20_last.json`
- `scout_eval_compare_seed3407_stop20_best_vs_last.json`
- `scout_eval_bucket_analysis_seed3407_stop20_best.json`
- `scout_eval_bucket_analysis_seed3407_stop20_last.json`
- `scout_eval_per_image_seed3407_stop20_best.csv`
- `scout_eval_per_image_seed3407_stop20_last.csv`
- `scout_eval_per_image_seed3407_stop20_best_vs_last.csv`

No checkpoints, images, arrays, datasets, or raw inference outputs are part of
the synced evidence package.
