# ConvIR-B Execution Guide

Date: 2026-07-12

Status: ConvIR-B project overlay for the generic experiment protocol.

## Purpose

Use this guide when the generic `experience_docx` package is applied to this
repository. It connects the experiment process to the official ConvIR-B
checkpoints, task entrypoints, metrics, and fixed-budget decision gates.

This guide defines the baseline and fixed-budget contracts for ConvIR-B work.
Do not modify the model until the official or repository-provided pretrained
checkpoint has been evaluated on the authorized runtime host and any
reproduction gap has a written explanation. Local WSL remains editing and
syntax/static-check only.

## Required Order

1. Resolve the official pretrained model for the target task from the root
   `README.md` model links; reuse an existing verified asset when its hash and
   identity match instead of downloading it again.
2. Record the checkpoint source, runtime path, file size, and sha256 hash.
3. Run the repository evaluation command for the target task and dataset on the
   authorized runtime host.
4. Record the exact data/metric view, verified sample count, and aggregate PSNR
   and SSIM when available.
5. Compare the reproduced cloud result with the official table in the root
   `README.md`.
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
exact cloud runtime command in the experiment card:

| Task | Entrypoint | ConvIR-B note |
| --- | --- | --- |
| Motion deblurring GoPro | `Motion_Deblurring/main.py --mode test` | fixed large-style entrypoint by default; use as task baseline only after variant verification |
| Deraining Test100/Test2800 | `Image_deraining/test.py` or task README flow | fixed large-style entrypoint by default; score calculation may require the repository's external MATLAB step |
| Dehazing ITS/Haze4K/NHR/GTA5 | `Dehazing/ITS/main.py --mode test --version base` | choose the `--data` value that matches the checkpoint |
| Dehazing OTS | `Dehazing/OTS/main.py --mode test --type base` | this folder uses `--type`, not `--version` |

## Baseline Record Fields

Every baseline reproduction note must include enough to reproduce the aggregate
claim:

- dataset root and verified sample count;
- checkpoint source URL;
- runtime checkpoint path;
- checkpoint sha256;
- checkpoint file size;
- git commit or source snapshot;
- explicit Python plus material framework/runtime and GPU identity;
- command line and working directory;
- PSNR and SSIM when the task code reports them;
- difference from the official table;
- whether the gap is accepted, explained, or blocking.

Add latency/memory only when a later cost gate uses them. Add raw-output,
per-sample-table, image, or visual-review paths only when they are generated for
a written mechanism, failure-diagnosis, safety, or promotion gate. These raw
artifacts stay on cloud.

## Replacement Budget Contract

The default route question is:

```text
Can the candidate beat the matched ConvIR-B runtime baseline under the same data,
evaluation, and hardware contract while staying within the cost limits?
```

The following values are a drop-in replacement reference, not universal route
eligibility. Use them only when the route objective is an operational ConvIR-B
replacement and no newer deployment budget exists. Diagnostic, mechanism, and
positive-ablation routes may exceed them, but cannot claim drop-in replacement.

| Constraint | Default limit |
| --- | --- |
| FLOPs | <= ConvIR-B FLOPs + 5% |
| Parameters | record always; no increase accepted unless the card explains why |
| Peak GPU memory | <= matched runtime baseline peak memory + 10% and must fit the current GPU |
| Average latency | <= matched runtime baseline average latency + 10% |
| Inference output size | same as baseline |
| Checkpoint/export/resume | same contract unless explicitly tested |

Select the route profile with `MODEL_EXPERIMENT_START_CHECKLIST.md`. For a
training route, the route card defines its short scout, first hard gate, and
formal decision budget from the matched baseline and the earliest decisive
question. `audit_evaluation` and `policy_replay` profiles do not inherit
training epoch stages.

Use the same declared budget points for the baseline learning curve when
training is part of the comparison. A candidate cannot be called faster unless
it is compared with the matched baseline at the same epoch, step, or wall-clock
budget.

## Formal Decision Metrics

Do not judge a formal image-restoration decision only by average PSNR. Retain
the smallest aggregate set that supports the route's typed gate:

- primary quality effect with grouped uncertainty at the claim's analysis unit;
- protected/strong-case and lower-tail regression summaries;
- the one mechanism metric required by the route hypothesis;
- latency, memory, or FLOPs only when cost is part of the gate.

Generate raw per-image deltas when needed for these aggregates, but keep them in
cloud `RUN_ROOT`. Do not curate a large per-image table, visual artifact catalog,
FFT report, or edge/texture report at every stage. Add those only for a relevant
mechanism claim, a diagnosed failure, or a terminal promotion audit.

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
| early scientific gate fails on quality and mechanism | stop the written continuation; inspect target definition before more budget |
| scientific utility fails on quality but mechanism moves | keep as a typed ablation or run only a predeclared cheaper diagnostic |
| scientific utility fails on mechanism | close the current hypothesis, not unrelated route families |
| safety/promotion fails on regressions | block the written promotion; do not infer that the mechanism is absent |
| final cost fails | keep as ablation only unless the project explicitly changes deployment constraints |

## Do Not Globalize Route Parameters

Checkpoint filenames, artifact roots, seed sets, model modifications, sample
sizes, and route-specific thresholds belong in the route card. Do not turn one
route's convenient values into repository-wide defaults.
