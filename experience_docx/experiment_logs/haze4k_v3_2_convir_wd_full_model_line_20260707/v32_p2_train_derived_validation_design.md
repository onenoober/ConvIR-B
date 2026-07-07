# Haze4K v3.2 ConvIR-WD P2 Train-Derived Validation Design

Status: `P2_DESIGN_WRITTEN_LOCKED_TEST_BLOCKED`.

## Fact Sources

- GitHub `main`: `633fbcc`.
- GitHub route branch: `codex/haze4k-v3-2-convir-wd-full-model-line` at
  `4e7554f` before P2 design code.
- Runtime host: `convir-4090`.
- Runtime workspace:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3-2-convir-wd`.
- Python:
  `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Haze4K data:
  `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- Official A0 checkpoint:
  `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- v3.1 raw 600-image train-derived table, cloud-only:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-github-main/experience_docx/experiment_logs/haze4k_v3_1_full_model_candidate_bakeoff_20260707/v31_candidate_per_image_cloud_only.csv`.

## Route Identity

P2 is a continuation of the v3.2 ConvIR-WD full-model line after P0 and P1b
passed. It is not a v3.0 rescue, not an A0 residual route, not a selector/alpha
route, and not a bridge/generator route.

Forbidden in P2:

- locked Haze4K test access;
- canary80 as a shortcut;
- threshold/checkpoint selection from locked test;
- changing scope, split, seed, loss weights, or checkpoint policy after seeing
  validation results.

## Split Contract

P2 uses the same 600 train-derived image names as v3.1 so ConvIR-WD can be
compared with the v3.1 standalone candidate table without touching locked test.

- validation split: `fold_id=0`, 120 images;
- training split: `fold_id=1..4`, 480 images;
- source images: original Haze4K `train/haze` and `train/gt` only;
- derived data root:
  `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K_v32_p2_fold0_train480_val120`;
- split construction: symlinked files only; original Haze4K `test` is not read
  or linked.

The split script writes a compact `v32_p2_split_summary.json` with counts and
hashes, not a full per-image table.

## Training Contract

Fixed run:

- model name:
  `ConvIR-Haze4K-v32-p2-wddecoder-seed3407-20260707`;
- architecture: `convir_wd_lite`;
- init: strict partial-load from official A0 checkpoint;
- scope: `wd_decoder`;
- seed: `3407`;
- epochs: `20`;
- batch size: `4`;
- WD LR: `2e-4`;
- decoder LR: `1e-5`;
- grad clip: `0.01`;
- save/validation frequency: every `5` epochs;
- modulation stats: every `5` epochs over up to `32` validation batches.

Checkpoint policy:

- primary P2 checkpoint: `Best.pkl`, selected only by P2 validation PSNR;
- supportive checkpoint: `Final.pkl`;
- no hyperparameter, scope, checkpoint, or threshold changes after seeing P2
  validation metrics under this run id.

## Metric Contract

Baseline: official ConvIR-B A0 on the exact same 120 validation images and same
full-image padding/evaluation code.

Primary P2 gate for `Best.pkl`:

- mean PSNR delta vs A0 `>= +0.30 dB`;
- hard bottom-25% PSNR delta vs A0 `>= +0.50 dB`;
- easy top-25% PSNR delta vs A0 `>= -0.05 dB`;
- p05 PSNR delta `>= -0.30 dB`;
- CVaR5 PSNR delta `>= -0.50 dB`;
- mean SSIM delta `>= -0.001`;
- catastrophic proxy count is `0`, where catastrophic means PSNR delta
  `<= -2.0 dB` or SSIM delta `<= -0.02`;
- Pareto-competitive with v3.1 standalone candidates on the same 120 names:
  ConvIR-WD must not be dominated by WDMamba or ConvIR-L across mean, hard,
  easy, p05, and CVaR5 PSNR deltas.

P2 pass authorizes writing a P3 fixed internal confirmation design only. P2
does not authorize locked test access by itself.

P2 fail stops this v3.2 `wd_decoder` screen. Do not rescue it by changing
epochs, scope, fold, learning rates, or loss weights under the same run.

## Artifact Boundary

Commit compact text evidence only:

- design markdown;
- durable scripts;
- split summary JSON;
- train/eval logs;
- P2 summary/gate JSON;
- route card, README, closeout, central index, and family summary.

Do not commit checkpoints, model weights, image outputs, datasets, symlinked
data roots, raw inference outputs, or full per-image P2 tables by default.
