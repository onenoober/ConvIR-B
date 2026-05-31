# Haze4K FAM Feature Modulation

Date: 2026-05-31

Status: 5-epoch scout completed; unchanged `modres` is diagnostic, not promoted.

## Scope

- Project: ConvIR-B dehazing.
- Model family: ConvIR-B.
- Dataset or task: Haze4K image dehazing.
- Primary objective: test whether conditional FAM modulation improves Haze4K under the reproduced ConvIR-B baseline contract.
- Main metric: PSNR.
- Secondary metrics: SSIM, per-image PSNR delta, strong-case regression count, latency, peak GPU memory.
- Execution environment: `autodl-dehaze3`, RTX 5090, PyTorch `2.11.0+cu128`.
- Artifact root: `Dehazing/ITS/results/ConvIR-Haze4K-fam-modres-*`.
- Branch or isolated workspace: current working tree; do not stage unrelated docs or artifact logs with model code.
- Review package location: `experience_docx/baseline_logs/haze4k_pretrained_20260531/`.

## Baseline Contract

- Baseline implementation: `Dehazing/ITS/main.py` with `--version base --data Haze4K --fam_mode original`.
- Baseline checkpoint or initialization: official/repository checkpoint `haze4k-base.pkl`.
- Evaluation entrypoint: `Dehazing/ITS/main.py --mode test`.
- Training entrypoint: `Dehazing/ITS/main.py --mode train`.
- Dataset and split: `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`; test has 1000 hazy and 1000 GT images.
- Preprocessing and decoding: current Haze4K loader accepts `haze/gt`, filters image extensions, and maps names such as `1000_0.74_1.6.png -> 1000.png`.
- Metric implementation: PSNR and SSIM in `Dehazing/ITS/eval.py`.
- Reproduced baseline result: PSNR `34.14 dB`, SSIM `0.98971`, average time `0.083973 s/image`, peak GPU memory `1329 MiB`.
- Known reproduction gap: official ConvIR-B Haze4K table is `34.15 / 0.99`; local result matches within rounding tolerance.
- Reference entrypoints that must remain stable: default `--fam_mode original` must load the official checkpoint and reproduce the baseline.
- Checkpoint/export/resume contract: original checkpoints remain strict-load compatible when `--fam_mode original`.

## Most Valuable Attempt

- Why this is the highest-value next attempt: FAM is only two local fusion points and currently uses static `cat + 3x3 conv`; the change directly tests whether conditional fusion is a bottleneck.
- Target failure or opportunity: non-uniform haze regions may need SCM context to actively calibrate encoder features rather than being passively concatenated.
- Cheap preflight evidence: zero-init modulation should be exactly neutral when enabled; default mode should be byte-for-byte compatible with existing checkpoints.
- Earliest decisive gate: neutral-init output equality and finite train/eval smoke.
- Expected cost or attempt-count saving: reject the route before 5/20 epoch training if neutral behavior, cost, or checkpoint compatibility fails.
- What success decides: keep FAM modulation as the first architecture pilot and run matched 5-epoch scout.
- What failure decides: close the modulation variant and try the lower-risk FFL loss route before larger operators.
- Why a cheaper diagnostic is not enough: static analysis cannot show whether learned conditional modulation improves restoration without training.

## Hypothesis

- Observed failure: FAM only compresses concatenated main and SCM features.
- Target mechanism: learned conditional scale-and-shift over the fused feature.
- Primary variable: FAM fusion mode.

Mechanism sentence:

```text
If we add zero-initialized SCM-conditioned modulation after the original FAM
merge, Haze4K restoration should improve because low-resolution haze context can
actively calibrate the fused encoder feature without changing the default
baseline behavior.
```

## Change

- Code branch: current workspace, pending branch decision.
- Exact code/config change: add optional `--fam_mode {original,modres}`; `modres` computes original FAM output, then applies zero-initialized `1x1` gamma/beta modulation from SCM features. Add optional `--stop_epoch` so short scouts can keep the official `--num_epoch 1000` scheduler horizon while stopping early.
- Enabled mechanisms: SCM-conditioned residual feature modulation.
- Explicitly disabled mechanisms: no FFL, no DeepPool/DCNv4, no data augmentation change, no optimizer change.
- Parameter/runtime/memory impact expected: about `41,344` extra parameters for two 1x1 modulators, roughly `+0.48%` over ConvIR-B; expected latency and memory increase below `+10%`.
- Initialization or no-op behavior: gamma/beta projection weights and bias are zero, so `modres` starts exactly as original FAM.
- Resume policy: do not resume baseline checkpoints into `modres` unless the loader explicitly handles missing modulation keys; first pilot is from scratch.
- Defaults changed: none.
- Defaults intentionally preserved: `--fam_mode original`, `--seed -1`, and `--stop_epoch -1` are defaults.

## Preflight

| Check | Pass line | Result |
| --- | --- | --- |
| shape/static check | original and `modres` build and emit three outputs with baseline shapes | pass on `256x256`; `128x128` is invalid for baseline DeepPool reflect padding |
| finite forward/backward | one fixed batch produces finite loss and gradients | pass with synthetic batch size 2 and real Haze4K batch size 8 |
| neutral-init or no-op | after copying shared weights, original and `modres` outputs match within `1e-6` max abs diff | pass; max diffs `[0.0, 0.0, 0.0]` |
| small overfit or probe | one train step completes with `--fam_mode modres` | pass; one Haze4K train batch, total loss `1.3300607` |
| cost check | parameter increase <= `+1%`; measured eval latency and peak memory <= `+10%` before promotion | static pass; `+41,344` params, `+0.479%`; eval latency/memory pending |

## Mechanism Metrics

| Metric | Why it matches the route | Gate subset | Final artifact |
| --- | --- | --- | --- |
| modulation gamma/beta mean and std | verifies the added path learns non-zero conditioning | validation subset after training | train/eval log or analysis note |
| FAM2-only vs FAM1+FAM2 ablation | separates half-resolution and quarter-resolution fusion value | scout runs | experiment log |
| per-image PSNR delta vs baseline | catches average-score wins that regress many images | full Haze4K test when feasible | CSV or summary table |
| strong-reference regression count | protects images already handled by baseline | top 25% baseline PSNR group | regression list |
| latency and peak GPU memory | enforces fixed-budget comparison | timed eval subset/full eval | run log |

## 5-Epoch Scout Evidence

- Run root: `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/ITS/results/ConvIR-Haze4K-fam-modres-scout-20260531-stop5`.
- Local evidence copy: `experience_docx/experiment_logs/haze4k_fam_modres_scout_stop5_20260531/`.
- Matched command policy: both runs used `--num_epoch 1000 --stop_epoch 5 --batch_size 8 --learning_rate 4e-4 --valid_freq 1 --seed 3407`; this preserves the official scheduler horizon and stops early.
- Original train-validation PSNR by epoch: `20.60`, `20.64`, `21.37`, `21.90`, `22.31`.
- `modres` train-validation PSNR by epoch: `20.77`, `20.58`, `20.54`, `22.40`, `21.94`.
- Epoch-5 train-validation gap: `21.94 - 22.31 = -0.37 dB`; this is inside the written scout tolerance of `-0.50 dB`.
- Full-test check on each run's 5-epoch `Best.pkl`: original `22.3070 PSNR / 0.92093 SSIM`; `modres` `22.4023 PSNR / 0.91905 SSIM`.
- Full-test mean delta: `+0.0953 dB PSNR`, but mean SSIM delta is `-0.00187`.
- Delta distribution: median `-0.0490 dB`, p5 `-3.3088 dB`, p95 `+4.0541 dB`, worst-10% mean `-3.4940 dB`, best-10% mean `+4.1806 dB`.
- Preservation screen: top-quartile original-PSNR cutoff `24.9883 dB`; strong-reference regressions with delta <= `-0.05 dB` are `142/250`; all-image worst regressions with delta <= `-0.20 dB` are `469/1000`.
- Cost screen from synchronized full-test script: original avg `0.05279 s/image`, `570.69 MiB` peak allocated; `modres` avg `0.05196 s/image`, `572.05 MiB` peak allocated. This is within the `+10%` latency/memory budget, but absolute timing is from the comparison script and not directly comparable to the earlier pretrained baseline timing.
- Mechanism liveness: trained modulation parameters moved away from zero. `FAM1.modulator.weight` mean abs `0.00306372`, `FAM1.modulator.bias` mean abs `0.03165563`; `FAM2.modulator.weight` mean abs `0.00713783`, `FAM2.modulator.bias` mean abs `0.00507046`.
- Stability note: both original and `modres` had early loss spikes under the official warmup schedule; `modres` additionally shows a highly split per-sample effect, so average PSNR alone is misleading.

## Controls

| Control | Purpose | Pass line |
| --- | --- | --- |
| original FAM baseline | fair reference | reproduced pretrained baseline remains loadable |
| zero-init `modres` equality | prove implementation starts as no-op | max output diff <= `1e-6` after shared-weight copy |
| add-only route not included | keep one primary variable | no competing fusion variant in first scout |

## Fair Run Contract

- Training or inference budget: preflight, then 5 epoch scout before any 20 epoch run.
- Batch/sample policy: Haze4K training batch size 8 unless memory forces a written change.
- Optimizer: Adam, repo default betas and epsilon.
- Schedule: warmup + cosine, unchanged.
- Loss weights: original three-scale L1 plus `0.1 * FFT-L1`, unchanged.
- Random seed policy: record seed for every run; first pilot may use one seed, promoted claims need more seeds if feasible.
- Evaluation cadence: at least every epoch for smoke; every 5 epochs for scout unless runtime suggests otherwise.
- Checkpoint cadence: keep `Best.pkl`, `Final.pkl`, and logs for scout.
- Hardware/runtime assumptions: `autodl-dehaze3`, RTX 5090, `convir-cu128`.
- Allowed resume behavior: no cross-mode resume unless explicitly tested.
- Sample-size policy: full Haze4K validation/test for formal gates; small subsets only for preflight.
- Dependency/version assumptions: install missing dependencies directly and record durable facts.

## Gates

| Gate | Image/global metric rule | Mechanism rule | Stop/continue rule |
| --- | --- | --- | --- |
| sanity | output shapes match baseline; one batch train/eval is finite | neutral-init diff <= `1e-6` | stop and fix implementation if failed |
| early trajectory | 5-epoch PSNR is not more than `0.50 dB` below matched original scout | gamma/beta leaves zero after training | continue only if cost and regression checks are plausible |
| first hard gate | 20-epoch PSNR within `0.25 dB` of matched original or target-group gain is clear | modulation statistics are non-degenerate | promote only if strong-case regression <= `2%` and cost limits hold |
| promotion | 80-epoch mean PSNR >= matched original - `0.10 dB` | mechanism still supports hypothesis | continue to full only if regression/cost limits still hold |
| final | PSNR gain >= `+0.10 dB` and SSIM delta >= `-0.001` | controls do not contradict the claim | label positive only if quality, mechanism, preservation, and cost pass |

## Analysis Plan

- Per-sample or subgroup analysis: compare PSNR delta distribution against reproduced baseline.
- Visual or qualitative analysis: inspect residual haze, edge halo, color shift, and sky/remote-scene artifacts.
- Complexity analysis: parameter delta, eval latency, and peak GPU memory.
- Robustness or held-out analysis: consider NHR/GTA5 only after Haze4K scout passes.
- Regression analysis: top-baseline quartile strong-reference group and worst-case PSNR drops.
- Required docs to update: this card and experiment log/runbook.
- Required artifacts to retain: command, logs, checkpoints for promoted runs, compact metrics tables.
- Required artifacts to delete or keep external: raw images/checkpoints stay external to git.
- Evidence package contents: text logs, summaries, config, metric CSV if produced.
- Evidence package audit: compare source, local copy, and final evidence list before sharing.

## Decision

- Decision label: diagnostic positive mechanism, failed preservation screen; do not promote unchanged `modres` as a replacement route.
- Image/global metric reason: the epoch-5 train-validation gap is within the scout tolerance, and the full-test `Best.pkl` comparison gives a small mean PSNR gain of `+0.0953 dB`.
- Mechanism reason: modulation parameters moved away from zero, so the added path is active rather than a dead branch.
- Preservation or regression reason: the average gain is not trustworthy because median PSNR delta is negative, worst-10% mean delta is `-3.4940 dB`, strong-reference regressions are `142/250`, and all-image regressions <= `-0.20 dB` are `469/1000`.
- Cost/deployability reason: parameter delta is `+0.479%`; synchronized full-test latency and peak allocated memory are within the `+10%` scout budget.
- What this decides next: do not spend a 20-epoch replacement run on unchanged `modres`. Either narrow/protect the modulation route with an explicit no-regression mechanism, or move to the lower-risk FFL/loss route before larger architecture changes.

## Preflight Evidence

- `experience_docx/experiment_logs/haze4k_fam_modres_preflight_20260531/preflight_20260531-211215_crop256_batch2.log`
- `experience_docx/experiment_logs/haze4k_fam_modres_preflight_20260531/one_batch_train_probe_20260531-211256.log`

## Invalid Scout Attempt

- Command shape: `--num_epoch 5` with the repository scheduler.
- Result: invalid. The scheduler uses `T_max=args.num_epoch-warmup_epochs`; with `num_epoch=5`, epoch 4 LR jumped to about `0.001597`, original FAM loss exploded, and PSNR dropped to `17.29 dB`.
- Decision: do not use `--num_epoch 5` for Haze4K scout. Use `--num_epoch 1000 --stop_epoch 5` so the short run follows the official training schedule shape.

## 5-Epoch Scout Artifacts

- `experience_docx/experiment_logs/haze4k_fam_modres_scout_stop5_20260531/original_train_stop5_seed3407.log`
- `experience_docx/experiment_logs/haze4k_fam_modres_scout_stop5_20260531/modres_train_stop5_seed3407.log`
- `experience_docx/experiment_logs/haze4k_fam_modres_scout_stop5_20260531/scout_eval_compare_seed3407.json`
- `experience_docx/experiment_logs/haze4k_fam_modres_scout_stop5_20260531/scout_eval_per_image_seed3407.csv`
- `experience_docx/tools/eval_haze4k_checkpoint_compare.py`
