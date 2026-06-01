# Haze4K Stop20 Original Noise Floor

Date: 2026-06-01

Status: completed on AutoDL; this is the P0 noise-floor measurement before any
new gate budget tuning.

## Scope

- Project: ConvIR-B dehazing.
- Dataset or task: Haze4K image dehazing.
- Branch: `codex/haze4k-stop20-noise-floor`, based on the confidence-gate line.
- Primary objective: measure the seed and trajectory noise floor of matched
  20-epoch from-scratch `--fam_mode original` training.
- Frozen variables: original architecture, original loss, no FAM modulation,
  no gate budget, no optimizer change, no crop change, no schedule change.
- Artifact roots:
  - `experience_docx/experiment_logs/haze4k_stop20_noise_floor_20260601/`
  - `Dehazing/ITS/results/ConvIR-Haze4K-original-stop20-noise-floor-20260601/`

## Why This Is Next

The confidence-gated gamma run produced a positive Best checkpoint
(`+0.4523 dB` versus the locked seed-3407 original Best), but the gate had
collapsed to near zero before that Best checkpoint. That makes the result a
training-trajectory or checkpoint-selection confound rather than clean evidence
that an active gate solved the FAM2 selectivity problem.

Before spending more runs on gate penalties, measure whether the current
single-seed stop20 protocol is powered enough to support route decisions.

## Baseline Contract

- Reuse the existing matched seed-3407 original stop20 checkpoint if present:
  `Dehazing/ITS/results/ConvIR-Haze4K-original-stop20-seed3407-20260531/Training-Results/Best.pkl`.
- Train additional original seeds with the same command surface:
  `--data Haze4K --version base --fam_mode original --batch_size 8
  --learning_rate 4e-4 --num_epoch 1000 --stop_epoch 20 --valid_freq 1
  --save_freq 5`.
- Default seeds: `3407 2027 8675`.
- The only variable is training seed.

## Run Script

```bash
cd /root/autodl-tmp/workspace/ConvIR-B
bash experience_docx/experiment_logs/haze4k_stop20_noise_floor_20260601/run_original_multiseed_stop20.sh
```

Optional custom seeds:

```bash
SEEDS="3407 2027 8675 9011" \
bash experience_docx/experiment_logs/haze4k_stop20_noise_floor_20260601/run_original_multiseed_stop20.sh
```

The script trains missing seed checkpoints, reuses seed `3407` when available,
then evaluates all seed Best checkpoints on the full 1000-image Haze4K test set.
For newly trained seeds, completion is defined by `Final.pkl`, not `Best.pkl`;
an interrupted partial directory is archived before that seed is restarted from
scratch. This keeps scheduler and Best-checkpoint behavior comparable across
seeds.

## Produced Artifacts

| Artifact | Purpose |
| --- | --- |
| `original_train_stop20_seed*.log` | training curve for every newly trained seed |
| `status.txt` | launch, reuse, completion, and failure timestamps |
| `original_seed_noise_stop20.json` | overall and bucket-level seed noise summary |
| `original_seed_noise_per_image.csv` | per-image PSNR/SSIM by seed plus cross-seed std |

The evaluation tool is:

```bash
python experience_docx/tools/eval_haze4k_seed_noise.py \
  --data_dir /root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K \
  --mode original \
  --run 3407:/path/to/seed3407/Best.pkl \
  --run 2027:/path/to/seed2027/Best.pkl \
  --run 8675:/path/to/seed8675/Best.pkl \
  --reference_seed 3407 \
  --output_json /path/to/original_seed_noise_stop20.json \
  --output_csv /path/to/original_seed_noise_per_image.csv
```

## Result Summary

The completed run used seeds `3407 2027 8675`, with seed `3407` reusing the
matched locked original stop20 Best checkpoint and seeds `2027` and `8675`
trained from scratch. All seed Best checkpoints were evaluated on the full
1000-image Haze4K test set.

| Metric | Value |
| --- | ---: |
| mean PSNR across seeds | `24.7329 dB` |
| mean PSNR sample std | `0.2206 dB` |
| mean PSNR range | `0.4124 dB` |
| mean SSIM across seeds | `0.948654` |
| mean SSIM sample std | `0.001414` |
| per-image PSNR std mean / median / p95 | `1.1047 / 0.9610 / 2.4438 dB` |

Per-seed means:

| Seed | Mean PSNR | Mean SSIM |
| --- | ---: | ---: |
| `3407` | `24.6424` | `0.947803` |
| `2027` | `24.9844` | `0.950286` |
| `8675` | `24.5720` | `0.947872` |

Bucket-level PSNR noise:

| Bucket | Mean PSNR | Sample std | Range |
| --- | ---: | ---: | ---: |
| hard bottom 25% | `19.8574` | `0.4551` | `0.9062` |
| medium middle 50% | `24.7759` | `0.2268` | `0.4284` |
| easy top 25% | `29.5225` | `0.3240` | `0.6389` |

Interpretation: the global stop20 protocol is not stable enough for
single-seed `+0.3 dB` route decisions, and the hard/easy bucket variance is
large enough that bucket-specific claims need repeat seeds or a longer horizon.
The earlier confidence-gate `+0.4523 dB` result remains diagnostic evidence,
not promotion-grade evidence, until repeated seeds show the gain survives this
measured noise floor.

## Decision Gates

| Gate | Interpretation |
| --- | --- |
| overall mean PSNR sample std <= `0.10 dB` | stop20 original is stable enough that `+0.3` to `+0.45 dB` candidate gains deserve follow-up |
| overall mean PSNR sample std >= `0.30 dB` or range >= `0.60 dB` | stop20 single-seed candidate-vs-original decisions are underpowered |
| hard/easy bucket std differs strongly | future route cards must report bucket-specific noise, not only global PSNR |
| per-image PSNR std is large in hard bucket | hard bottom-25% gains need repeat seeds or a longer horizon before promotion |

## What Success Decides

- If original stop20 is stable, run the next gamma-only gate variant with an
  internally balanced penalty: target-budget or hard/easy gap hinge.
- If original stop20 is noisy, pause gate tuning and either use repeated seeds
  for every stop20 candidate or move the most justified variant to a longer
  horizon where the baseline variance is lower.
- In either case, do not treat the confidence-gate `+0.4523 dB` as promotion
  evidence until it is compared against this measured noise floor.
