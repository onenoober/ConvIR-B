# Haze4K Haze-Prior SCM

Date: 2026-06-01

Status: paired 20-epoch candidate/control scout completed on `autodl-dehaze3`.
Decision is no 80-epoch promotion for the exact `haze_prior SCM + hard_aux`
configuration because the Best checkpoint loses mean PSNR/SSIM to the matched
hard-aux control and the hard-bucket signal is bought with excessive
strong/easy regressions.

## Scope

- Project: ConvIR-B dehazing.
- Model family: ConvIR-B.
- Dataset or task: Haze4K image dehazing.
- Primary objective: replace ConvIR's SCM input-conditioning branch with a
  haze-prior-aware SCM while keeping FAM behavior original.
- Main metric: Haze4K test PSNR.
- Secondary metrics: SSIM, per-image PSNR delta, hard/mid/easy bucket delta,
  strong-reference regressions, worst-case regressions, latency, peak memory,
  parameter delta, SCM prior-branch activity.
- Execution environment: local WSL for edits/static/synthetic checks; AutoDL or
  documented cloud server for real Haze4K preflight, evaluation, and training.
- Artifact root: `experience_docx/experiment_logs/haze4k_haze_prior_scm_20260601/`.
- Branch or isolated workspace: `codex/haze4k-haze-prior-scm` in
  `/home/ubuntu/workspace/ConvIR-B-haze-prior-scm`.
- Review package location: pending after first cloud evidence package.

## Baseline Contract

- Baseline implementation: `Dehazing/ITS/main.py --data Haze4K --version base
  --fam_mode original --scm_mode original`.
- Candidate implementation: same entrypoint with `--scm_mode haze_prior`.
- Baseline checkpoint or initialization: matched from-scratch Haze4K training
  unless a card records a checkpoint-compatible evaluation.
- Evaluation entrypoint: `Dehazing/ITS/main.py --mode test`.
- Training entrypoint: `Dehazing/ITS/main.py --mode train`.
- Dataset and split: Haze4K train/test from the documented cloud dataset root.
- Preprocessing and decoding: repository `Dehazing/ITS/data` loaders and
  evaluation padding/cropping.
- Metric implementation: repository PSNR/SSIM plus per-image comparison tools.
- Reproduced baseline result: use the existing stop20 original noise-floor
  card as the short-horizon variance reference; do not claim success from a
  single-seed small gain.
- Known reproduction gap: none added by this card.
- Reference entrypoints that must remain stable: default `--fam_mode original
  --scm_mode original --loss_mode original` must keep the original model
  behavior and strict checkpoint contract.
- Checkpoint/export/resume contract: candidate checkpoints require
  `--scm_mode haze_prior`; original checkpoints use the default SCM.

## Most Valuable Attempt

- Why this is the highest-value next attempt: the FAM selector/modulation route
  is closed by the selectivity-or-kill evidence, while SCM is still the native
  ConvIR location for multi-scale input conditions.
- Target failure or opportunity: ConvIR-B may need dehazing-specific condition
  features for density, low contrast, dark channel, and saturation without
  adding a deployable per-image selector.
- Cheap preflight evidence: zero-initialized prior branch must make candidate
  outputs equal original outputs at initialization, and the prior branch must
  receive finite gradients on a real Haze4K batch.
- Earliest decisive gate: 20-epoch hard gate, interpreted against the measured
  stop20 seed noise floor.
- Expected cost or attempt-count saving: avoids more FAM gate variants and
  tests a low-parameter branch before larger operators or FFL.
- What success decides: Haze-prior SCM becomes the next architecture candidate
  for 80-epoch and repeated-seed validation.
- What failure decides: close this SCM prior insertion or keep it diagnostic;
  only then consider hard-aware loss-only or FFL as the next candidate.
- Why a cheaper diagnostic is not enough: prior maps can be computed cheaply,
  but the claim is whether ConvIR learns to use them inside the SCM/FAM
  condition path without harming strong-reference images.

## Hypothesis

- Observed failure: FAM modulation has hard-sample upside but no deployable
  selector and large strong/easy regressions.
- Target mechanism: enrich SCM condition features with haze priors while FAM
  remains the original merge operator.
- Primary variable: SCM condition source.

Mechanism sentence:

```text
If we add zero-initialized haze-prior features to SCM, hard-haze and low-contrast
images should improve because the original multi-scale condition branch receives
dark-channel, saturation, brightness, and gradient cues without a per-image gate.
```

## Change

- Code branch: `codex/haze4k-haze-prior-scm`.
- Exact code/config change: add `--scm_mode {original,haze_prior}`. In
  `haze_prior`, each SCM becomes `SCM_rgb(x) + PriorBranch(prior_maps(x))`.
- Prior maps: min RGB, max RGB, dark channel, saturation, and grayscale
  gradient magnitude.
- Enabled mechanisms: zero-initialized haze-prior SCM branch and optional
  SCM-prior activity logging.
- Explicitly disabled mechanisms: FAM1 and FAM2 modulation, FAM gates,
  selector/router logic, optimizer changes, crop changes, schedule changes,
  data changes, larger attention blocks, and FFL in the first candidate.
- Training target for first candidate: `--loss_mode hard_aux` with a matched
  original-SCM hard-aux control before any SCM-specific claim.
- FFL status: second candidate only; blocked until this card has a decision.
- Parameter/runtime/memory impact expected: small parameter increase from the
  two prior branches; FLOPs/latency must remain within ConvIR-B fixed-budget
  limits.
- Initialization or no-op behavior: final prior-branch conv is zero-initialized,
  so candidate outputs must match original outputs at initialization.
- Resume policy: resume only within the same `scm_mode`, `fam_mode`, and
  `loss_mode`.
- Defaults changed: none.
- Defaults intentionally preserved: `--fam_mode original`, `--scm_mode
  original`, and `--loss_mode original`.

## Preflight

| Check | Pass line | Result |
| --- | --- | --- |
| shape/static check | Python compile passes for edited ITS files and preflight tool | pass locally with `py310` |
| neutral-init or no-op | synthetic same-seed candidate/original output max abs diff <= `1e-6` | pass; output diffs `[0.0, 0.0, 0.0]` |
| finite forward/backward | hard-aux batch loss finite; all gradients finite | pass locally; synthetic hard-aux total loss `10.4974699` |
| prior-branch trainability | prior branch has finite nonzero gradient | pass locally; nonzero prior grad params `10432/35776` |
| cost check | parameter delta recorded and comfortably below fixed-budget concern | pass locally; `+35776` params, `+0.4145%` |
| real-batch check | same checks pass on first Haze4K cloud batch | pass on `autodl-dehaze3`; output diffs `[0.0, 0.0, 0.0]`, total loss `1.5688771`, prior nonzero grad params `10432/35776`, peak CUDA memory `9392.02 MiB` |

Local synthetic preflight:

```bash
cd /home/ubuntu/workspace/ConvIR-B-haze-prior-scm
python3 experience_docx/tools/preflight_haze4k_haze_prior_scm.py \
  --batch_size 2 \
  --image_size 256 \
  --loss_mode hard_aux \
  --hard_aux_lambda 0.25 \
  --output experience_docx/experiment_logs/haze4k_haze_prior_scm_20260601/preflight_synthetic_seed3407.json
```

Cloud real-batch preflight:

```bash
cd /root/autodl-tmp/workspace/ConvIR-B
python3 experience_docx/tools/preflight_haze4k_haze_prior_scm.py \
  --data_dir /root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K \
  --batch_size 8 \
  --num_worker 0 \
  --loss_mode hard_aux \
  --hard_aux_lambda 0.25 \
  --output experience_docx/experiment_logs/haze4k_haze_prior_scm_20260601/preflight_real_batch_seed3407.json
```

## Mechanism Metrics

| Metric | Why it matches the route | Gate subset | Final artifact |
| --- | --- | --- | --- |
| SCM prior branch abs mean and prior/rgb ratio | verifies the new branch becomes active after neutral start | validation subset every logged gate | training log or JSON summary |
| prior-map summary means | confirms haze-prior inputs are finite and nondegenerate | preflight and gate subset | preflight JSON |
| hard/mid/easy PSNR delta | tests whether hard-haze gains survive without easy regression | full validation when feasible | per-image CSV |
| strong-reference regression count | protects images already handled by original ConvIR-B | top 25% baseline PSNR group | regression list |
| worst-case regression count | catches severe local failures | full validation | regression list |
| parameter, latency, and peak memory delta | enforces fixed-budget comparison | preflight and timed eval | run log |

## Controls

| Control | Purpose | Pass line |
| --- | --- | --- |
| original SCM + original loss | preserves the locked stop20 noise-floor reference | use existing card as variance floor |
| original SCM + hard_aux | separates SCM effect from hard-aware loss effect | required before claiming SCM-specific improvement |
| haze-prior SCM + original loss | optional ablation if hard_aux result is positive but attribution is unclear | use same budget before final claim |
| shuffled prior maps | later control if branch activity is high but image metrics are ambiguous | shuffled prior must not match real-prior gain |

## Fair Run Contract

- First candidate command difference: `--scm_mode haze_prior --loss_mode
  hard_aux --hard_aux_lambda 0.25`.
- FAM command: always `--fam_mode original`.
- Matched hard-aux control: same command with `--scm_mode original`.
- Training budget ladder: smoke, 5 epochs, 20 epochs, 80 epochs, full.
- Optimizer: unchanged Adam.
- Schedule: unchanged official `--num_epoch 1000` horizon with `--stop_epoch`
  for scouts.
- Random seed policy: seed `3407` for smoke/scout; repeat seeds before any
  small-gain claim because stop20 global std is `0.2206 dB` and hard-bucket std
  is `0.4551 dB`.
- Sample-size policy: full 1000-image Haze4K test at 20-epoch and later gates
  when feasible.

Candidate 20-epoch command template:

```bash
cd Dehazing/ITS
python main.py \
  --data Haze4K \
  --version base \
  --fam_mode original \
  --scm_mode haze_prior \
  --loss_mode hard_aux \
  --hard_aux_lambda 0.25 \
  --batch_size 8 \
  --learning_rate 4e-4 \
  --num_epoch 1000 \
  --stop_epoch 20 \
  --valid_freq 1 \
  --save_freq 5 \
  --seed 3407 \
  --scm_stats_freq 5 \
  --model_name ConvIR-Haze4K-hazeprior-scm-hardaux-stop20-seed3407-20260601
```

Matched hard-aux control uses the same command with:

```bash
--scm_mode original --model_name ConvIR-Haze4K-original-hardaux-stop20-seed3407-20260601
```

Cloud paired run script:

```bash
cd /root/autodl-tmp/workspace/ConvIR-B-haze-prior-scm
bash experience_docx/experiment_logs/haze4k_haze_prior_scm_20260601/run_haze_prior_scm_hardaux_stop20.sh
```

## Gates

| Gate | Image/global metric rule | Mechanism rule | Stop/continue rule |
| --- | --- | --- | --- |
| sanity | finite loss, output shapes equal baseline, compile passes | neutral-init max diff <= `1e-6`; prior branch has nonzero finite gradients | stop if any fail |
| early trajectory | 5-epoch PSNR within `0.50 dB` of matched control | SCM prior branch remains finite and starts nondegenerate activity | stop unless failure identifies an implementation fix |
| first hard gate | 20-epoch PSNR within `0.25 dB` of matched control or clear hard-bucket gain | hard-bucket gain is not bought by strong-reference regression > 2% | promote only if cost and preservation pass |
| promotion | 80-epoch mean PSNR >= matched control - `0.10 dB` | prior branch activity supports the hypothesis | continue only if regressions remain controlled |
| final | repeated-seed or full-budget gain exceeds noise-aware floor and SSIM delta >= `-0.001` | shuffled/basic controls do not explain the gain | label positive only if quality, mechanism, preservation, and cost all pass |

## Analysis Plan

- Per-sample or subgroup analysis: full per-image PSNR delta, hard/mid/easy
  buckets, top-25% strong-reference group, worst-case regressions.
- Visual or qualitative analysis: saved hard-bucket wins, easy regressions,
  color shift, halos, blur, residual haze.
- Complexity analysis: parameter count from preflight; FLOPs/latency before
  promotion.
- Robustness or held-out analysis: repeated seeds or longer horizon before
  claiming small gains.
- Regression analysis: compare against both original-loss noise floor and
  original-SCM hard-aux control.
- Required docs to update: this card, run status logs, and a text evidence
  package after cloud completion.
- Required artifacts to retain: scripts, JSON summaries, per-image CSV, concise
  logs.
- Required artifacts to keep external: raw checkpoints and image dumps unless a
  curated small text package is prepared.

## 20-Epoch Scout Evidence

Run location:

```text
/root/autodl-tmp/workspace/ConvIR-B-haze-prior-scm
```

Local evidence mirror:

```text
/home/ubuntu/workspace/ConvIR-B-haze-prior-scm/experience_docx/experiment_logs/haze4k_haze_prior_scm_20260601/
```

Completion:

- Remote tmux session ended; GPU was idle at verification time.
- `status.txt` ended with `complete 2026-06-01T18:39:12+08:00`.
- Generated evidence: synthetic preflight JSON, real-batch preflight JSON,
  matched original-SCM hard-aux train log, haze-prior-SCM hard-aux train log,
  Best/Last compare JSON, Best/Last bucket JSON, and Best/Last per-image CSV.

Training trajectory:

| Run | Epoch PSNR curve | Best | Last | Notes |
| --- | --- | --- | --- | --- |
| original SCM + hard_aux | 20.56, 20.85, 20.42, 22.23, 22.15, 22.84, 22.19, 23.13, 23.24, 21.35, 23.23, 23.38, 24.05, 24.94, 24.04, 23.73, 24.58, 23.55, 24.86, 22.36 | epoch 14, 24.94 dB | epoch 20, 22.36 dB | loss spikes at epoch 13 |
| haze-prior SCM + hard_aux | 20.38, 21.30, 20.18, 22.15, 22.54, 23.18, 23.01, 22.44, 22.58, 23.32, 23.84, 24.16, 23.77, 24.29, 22.54, 24.56, 23.80, 24.13, 23.84, 23.95 | epoch 16, 24.56 dB | epoch 20, 23.95 dB | loss spikes at epoch 16 |

Full-test Best checkpoint comparison against matched hard-aux control:

| Metric | original SCM + hard_aux | haze-prior SCM + hard_aux | Delta |
| --- | ---: | ---: | ---: |
| Mean PSNR | 24.935965 | 24.557032 | -0.378932 |
| Mean SSIM | 0.948732 | 0.943324 | -0.005407 |
| Positive delta ratio | n/a | 0.434 | n/a |
| Strong-reference regressions, delta <= -0.05 | n/a | 185 / 250 | n/a |
| Regressions, delta <= -0.20 | n/a | 528 / 1000 | n/a |

Best checkpoint bucket deltas:

| Bucket by original PSNR | Mean PSNR delta | Median PSNR delta | Positive ratio | Regressions <= -0.20 | Mean SSIM delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| hard bottom 25% | +0.350125 | +0.269632 | 0.556 | 100 / 250 | +0.000909 |
| medium middle 50% | -0.107365 | -0.182564 | 0.462 | 248 / 500 | -0.005532 |
| easy top 25% | -1.651124 | -1.392050 | 0.256 | 180 / 250 | -0.011474 |

Full-test Last checkpoint comparison:

| Metric | original SCM + hard_aux | haze-prior SCM + hard_aux | Delta |
| --- | ---: | ---: | ---: |
| Mean PSNR | 22.362059 | 23.946890 | +1.584831 |
| Mean SSIM | 0.913947 | 0.947844 | +0.033897 |
| Positive delta ratio | n/a | 0.747 | n/a |
| Strong-reference regressions, delta <= -0.05 | n/a | 93 / 250 | n/a |
| Regressions, delta <= -0.20 | n/a | 229 / 1000 | n/a |

Last checkpoint bucket deltas:

| Bucket by original PSNR | Mean PSNR delta | Median PSNR delta | Positive ratio | Regressions <= -0.20 | Mean SSIM delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| hard bottom 25% | +2.179099 | +2.069413 | 0.868 | 28 / 250 | +0.046059 |
| medium middle 50% | +1.668240 | +1.553973 | 0.756 | 111 / 500 | +0.030664 |
| easy top 25% | +0.823746 | +0.769413 | 0.608 | 90 / 250 | +0.028202 |

Mechanism and cost:

- Neutral-init behavior passed on synthetic and real Haze4K batches with output
  diffs `[0.0, 0.0, 0.0]`.
- Prior branch received finite nonzero gradients on real Haze4K batch:
  `10432/35776` prior-branch parameters had nonzero gradient.
- Parameter delta stayed small: `+35776` parameters, `+0.4145%`.
- SCM prior branch became active during training. Epoch 20 prior/rgb abs ratios:
  SCM1 `0.6943`, SCM2 `0.4843`.

Gate interpretation:

- Sanity gate passed.
- First hard gate did not pass promotion. Best checkpoint hard bucket improved
  by `+0.3501 dB`, but global mean was `-0.3789 dB`, SSIM was `-0.0054`, strong
  regressions were `185/250`, and easy top-25% mean delta was `-1.6511 dB`.
- Last checkpoint is not promotion evidence by itself because the matched
  original-SCM hard-aux final checkpoint collapsed to `22.3621 dB`; even there,
  strong-reference regressions remained high at `93/250`.
- Because the existing stop20 protocol has a global seed-noise std of
  `0.2206 dB` and hard-bucket std of `0.4551 dB`, this single-seed scout should
  be treated as diagnostic rather than a success claim.

## Decision

- Decision label: `NO_PROMOTE_STOP20_HAZE_PRIOR_SCM_HARDAUX`.
- Image/global metric reason: Best checkpoint underperforms the matched
  original-SCM hard-aux control by `-0.3789 dB` PSNR and `-0.0054` SSIM.
- Mechanism reason: the prior branch is active and trainable, so the insertion
  point is mechanically valid, but the learned effect is not preservation-safe.
- Preservation or regression reason: Best checkpoint has `185/250`
  strong-reference regressions at delta <= `-0.05`, `528/1000` regressions at
  delta <= `-0.20`, and easy top-25% mean delta `-1.6511 dB`.
- Cost/deployability reason: parameter cost is acceptable, but quality
  preservation fails the promotion gate.
- What this decides next: do not promote this exact `haze_prior SCM + hard_aux`
  configuration to 80 epochs. Keep the evidence as diagnostic hard-bucket
  signal; any next run should either isolate SCM without hard_aux or move to the
  next documented loss candidate with a matched original-SCM control and the
  same preservation gates.
