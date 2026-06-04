# ConvIR-B Execution Guide

Date: 2026-05-31

Status: ConvIR-B project overlay for the generic experiment protocol.

## Purpose

Use this guide when the generic `experience_docx` package is applied to this
repository. It connects the experiment process to the official ConvIR-B
checkpoints, task entrypoints, metrics, and fixed-budget decision gates.

The current phase is baseline establishment. Do not modify the model until the
official or repository-provided pretrained checkpoint has been evaluated in the
local environment and any reproduction gap has a written explanation.

Current runtime overlay for this checkout: the WSL copy is for editing and
compile/static checks only. Run smoke tests, training, evaluation, inference,
and other runtime validation on the configured cloud server. In older notes,
"local baseline" means the baseline measured in the active runtime environment,
not necessarily the WSL machine.

## Required Order

1. Download the official pretrained model for each target task from the root
   `README.md` model links.
2. Record the checkpoint source, local path, file size, and sha256 hash.
3. Run the repository evaluation command for the target task and dataset.
4. Record local baseline PSNR, SSIM when available, per-image PSNR, average
   latency, peak GPU memory, output path, and qualitative artifact notes.
5. Compare local results with the official table in the root `README.md`.
6. Explain any reproduction gap before starting a model-change route.
7. Optimize for best result under a fixed budget, not for an unconstrained
   "best score".

## Repository Facts To Anchor Cards

| Fact | Value or source |
| --- | --- |
| Main baseline family | ConvIR-B |
| Official ConvIR-B size | 8.63M parameters, 71.22G FLOPs |
| ConvIR-B desnowing CSD reference | 39.10 PSNR, 0.99 SSIM |
| ConvIR-B desnowing SRRS reference | 32.39 PSNR, 0.98 SSIM |
| ConvIR-B desnowing Snow100K reference | 33.92 PSNR, 0.96 SSIM |
| Desnowing train crop | 256 random crop in `Image_desnowing/data/data_load.py` |
| Desnowing train batch | default `--batch_size 8` |
| Desnowing validation/test split | `test2000`; verify actual file count before a formal card |
| Desnowing eval batch | batch size 1 inside `Image_desnowing/eval.py` |
| Desnowing metric code | PSNR from `skimage.metrics`, SSIM from `pytorch_msssim` |
| Desnowing padding | reflect pad to multiple of 32, then crop back |
| Official checkpoint source | root `README.md` pretrained model links |

If a repository README example uses `--version small`, replace it with
`--version base` for a ConvIR-B card and use the matching base checkpoint. If
the downloaded checkpoint naming is ambiguous, record the actual filename and
the model variant verified by successful loading.

Not every task folder exposes a base switch. `Image_deraining` and
`Motion_Deblurring` currently build a fixed `num_res=16` model, and the root
result table reports large-style rows for those tasks. Do not call those runs
ConvIR-B unless a base checkpoint and base-compatible entrypoint have been
verified.

## Baseline Evaluation Commands

Record the exact command that ran, not only the template.

### Desnowing CSD ConvIR-B

```bash
cd Image_desnowing
python main.py \
  --data CSD \
  --version base \
  --save_image True \
  --mode test \
  --data_dir <DATA_ROOT>/CSD \
  --test_model <CKPT_ROOT>/desnowing/<CSD_CONVIR_B_CHECKPOINT>.pkl
```

### Desnowing SRRS ConvIR-B

```bash
cd Image_desnowing
python main.py \
  --data SRRS \
  --version base \
  --save_image True \
  --mode test \
  --data_dir <DATA_ROOT>/SRRS \
  --test_model <CKPT_ROOT>/desnowing/<SRRS_CONVIR_B_CHECKPOINT>.pkl
```

### Desnowing Snow100K ConvIR-B

```bash
cd Image_desnowing
python main.py \
  --data Snow100K \
  --version base \
  --save_image True \
  --mode test \
  --data_dir <DATA_ROOT>/Snow100K \
  --test_model <CKPT_ROOT>/desnowing/<SNOW100K_CONVIR_B_CHECKPOINT>.pkl
```

### Other Target Tasks

Use the corresponding task README as the command authority, then record the
local command in the experiment card:

| Task | Entrypoint | ConvIR-B note |
| --- | --- | --- |
| Motion deblurring GoPro | `Motion_Deblurring/main.py --mode test` | fixed large-style entrypoint by default; use as task baseline only after variant verification |
| Deraining Test100/Test2800 | `Image_deraining/test.py` or task README flow | fixed large-style entrypoint by default; score calculation may require the repository's external MATLAB step |
| Dehazing ITS/Haze4K/NHR/GTA5 | `Dehazing/ITS/main.py --mode test --version base` | choose the `--data` value that matches the checkpoint |
| Dehazing OTS | `Dehazing/OTS/main.py --mode test --type base` | this folder uses `--type`, not `--version` |

## Baseline Record Fields

Every baseline reproduction note must include:

- dataset root and verified sample count;
- checkpoint source URL;
- local checkpoint path;
- checkpoint sha256;
- checkpoint file size;
- git commit or source snapshot;
- Python, PyTorch, CUDA, GPU model, driver, and cuDNN state;
- command line and working directory;
- PSNR and SSIM when the task code reports them;
- per-sample PSNR CSV path;
- average latency after warmup and number of timed images;
- peak GPU memory;
- output image directory;
- qualitative artifact count and examples by filename;
- difference from the official table;
- whether the gap is accepted, explained, or blocking.

## Fixed-Budget Contract

The default route question is:

```text
Can the candidate beat the local ConvIR-B baseline under the same data,
evaluation, and hardware contract while staying within the cost limits?
```

Default cost limits for a ConvIR-B replacement route:

| Constraint | Default limit |
| --- | --- |
| FLOPs | <= ConvIR-B FLOPs + 5% |
| Parameters | record always; no increase accepted unless the card explains why |
| Peak GPU memory | <= local baseline peak memory + 10% and must fit the current GPU |
| Average latency | <= local baseline average latency + 10% |
| Inference output size | same as baseline |
| Checkpoint/export/resume | same contract unless explicitly tested |

Default training-budget ladder:

| Stage | Budget | Purpose |
| --- | --- | --- |
| smoke | 0 to 1 epoch or fixed-batch probe | verify code, loss, gradients, and checkpoint path |
| scout | 5 epochs | reject collapsed or clearly expensive routes |
| first hard gate | 20 epochs | decide whether the route deserves meaningful training |
| promotion | 80 epochs | decide whether full training is informative |
| final | full budget from the card | make the replacement or ablation decision |

Use the same budget ladder for the baseline learning curve when training is part
of the comparison. A candidate cannot be called faster unless it is compared
with the matched baseline point at the same epoch, step, or wall-clock budget.

Interpret small deltas against the route-specific noise floor. For current
Haze4K stop20 work, `EXPERIMENT_INDEX.md` records seed mean PSNR std
`0.2206 dB` and hard-bucket std `0.4551 dB`; single-seed deltas below
`+0.10 dB` are directional or mechanism evidence by default, not promotion
evidence. Use internal validation, OOF, multi-seed, or locked confirmation
before treating such gains as candidate-positive.

## Default Gates For CSD Desnowing

These defaults are intentionally conservative and should be overwritten only
when a card gives a task-specific reason.

| Gate | Continue only if all are true |
| --- | --- |
| smoke | finite forward/backward; no NaN/Inf loss; output shape equals baseline; peak memory fits GPU; eval runs on at least 8 images |
| 5 epoch scout | PSNR is within 0.50 dB of the matched baseline scout point, latency <= +10%, no systematic artifact pattern in saved images |
| 20 epoch hard gate | PSNR is within 0.25 dB of the matched baseline 20-epoch point or shows a clear target-group gain; strong-case regression count <= 2% of validation images; peak memory <= +10% |
| 80 epoch promotion | mean PSNR is >= matched baseline - 0.10 dB and at least one mechanism metric supports the hypothesis; worst-10% PSNR delta is non-negative or explained |
| final replacement | mean PSNR gain >= +0.10 dB and SSIM delta >= -0.001; FLOPs <= +5%; latency <= +10%; strong-case regression count <= 1% |
| positive ablation | mechanism or target subgroup improves, but replacement gates are not met; label as ablation, not main baseline |

Minimum meaningful final improvement for CSD desnowing is `+0.10 dB PSNR` with
SSIM delta >= `-0.001`. Smaller gains can be retained only as
diagnostic evidence if they explain a mechanism or rule out a route.

## Always-On Mechanism And Regression Metrics

Do not judge image restoration only by average PSNR. At minimum, record these
for every formal ConvIR-B route:

- per-image PSNR and per-image delta versus the local ConvIR-B baseline;
- delta distribution: mean, median, worst 10%, best 10%, and p5/p95;
- strong-reference group: images where baseline PSNR is in the top 25%;
- strong-case regression count: strong-reference images with PSNR delta <=
  -0.05 dB;
- worst-case regression count: any image with PSNR delta <= -0.20 dB;
- edge or texture-region error summary if an analysis script exists;
- FFT loss or frequency-domain error summary when the route touches the loss,
  frequency blocks, or texture recovery;
- average latency, timed image count, warmup count, and peak GPU memory;
- output artifact count by simple labels such as color shift, ringing, blur,
  residual snow/rain/haze, edge halo, or texture washout.

Route-specific metrics are added only when relevant:

| Route type | Extra required mechanism evidence |
| --- | --- |
| selector, router, mask, or gate | entropy, selection distribution, false intervention on strong-reference images |
| preservation or no-regression guard | protected-case recall, guard activity, regression count |
| loss-only change | pixel-loss scale, FFT-loss scale, gradient norm health, target-group gain |
| architecture change | parameter/FLOP delta, latency delta, neutral-init or no-op behavior, branch activity |
| schedule or optimizer change | matched-step curve, time-to-threshold, stability, final quality |

## Failure Conclusions

Failure must still teach the next action.

| Failed gate | Default conclusion |
| --- | --- |
| baseline reproduction blocked | do not modify the model; fix data, checkpoint, metric, or environment contract |
| smoke fails | implementation invalid; debug shape, device, checkpoint, loss, or dependency path |
| 5 epoch scout fails on quality and mechanism | route is diagnostic only; inspect target definition before more training |
| 20 epoch hard gate fails on quality but mechanism moves | run a cheaper targeted ablation or adjust insertion/loss weight, not full training |
| 20 epoch hard gate fails on mechanism | close this route under the current hypothesis |
| 80 epoch promotion fails on regressions | add preservation guard or narrow target; do not promote as replacement |
| final cost fails | keep as ablation only unless the project explicitly changes deployment constraints |

## What Not To Fix Yet

Until the local baseline package exists, do not freeze:

- final checkpoint filenames beyond the recorded downloaded file;
- exact artifact root outside the current machine policy;
- final seed set beyond the default seed policy in the card template;
- a specific model modification route;
- route-specific thresholds for metrics that do not exist yet.

Those details become authoritative only after the first formal experiment card
is filled and the baseline reproduction evidence is attached.
