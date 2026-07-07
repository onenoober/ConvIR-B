# Haze4K v3.2 ConvIR-WD Full Model Line Evidence

Status: `COMPLETED_P0_P1B_AGGREGATE_PASS_P2_GATE_FAIL_LOCKED_TEST_BLOCKED`.

Route branch:
`codex/haze4k-v3-2-convir-wd-full-model-line`.

Runtime host: `convir-4090`.

Runtime paths:
- workspace:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3-2-convir-wd`;
- Python:
  `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`;
- data:
  `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`;
- official A0 checkpoint:
  `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.

Authorized stages:
- P0 architecture/preflight: completed and passed.
- P1 mini-overfit sanity: completed as a historical first-mini-batch gate.
- P1b aggregate mini-overfit sanity: completed and passed; this is the
  corrected aggregate P1 sufficiency check.
- P2 train-derived validation: completed; fixed gate failed.
- P3 fixed internal confirmation: not authorized from this route.
- P4 locked test: not authorized from this route.

Locked test policy: locked Haze4K test is blocked.

Primary outputs:
- `v32_p0_preflight.json`;
- `preflight_v32.log`;
- `v32_p1_mini_overfit.json`;
- `p1_mini_overfit_v32.log`;
- `v32_p1b_mini_overfit_aggregate.json`;
- `p1b_mini_overfit_aggregate_v32.log`;
- `run_p1b_mini_overfit_aggregate.sh`;
- `v32_p2_train_derived_validation_design.md`;
- `v32_p2_initial_engineering_failure.json`;
- `run_p2_train_derived_validation.sh`;
- `p2r1_train_v32_wddecoder_seed3407.log`;
- `p2r1_eval_best_v32.log`;
- `p2r1_eval_final_v32.log`;
- `v32_p2r1_eval_best_summary.json`;
- `v32_p2r1_eval_final_summary.json`;
- `v32_p2_closeout.json`;
- `status.txt`.
- `v32_closeout.json`.

## P0 Result

P0 ran on `convir-4090` from route commit `478ac83`.

- strict partial-load: `602` official keys loaded;
- allowed new WD keys: `24`;
- no-op max abs vs A0: `0.0`;
- one train-batch loss: `0.02780638262629509`;
- trainable manifest: `wd_only=309472`, `wd_decoder=4453433`,
  `all=8940137`;
- locked test touched: `false`.

Decision: `V32_P0_PREFLIGHT_OK`.

## P1 Result

P1 ran on `convir-4090` from route commit `35758db`.

- sample contract: `8` train-derived center crops, crop size `256`;
- train scope: `wd_decoder`;
- trainable params: `4453433`;
- initial/final loss: `0.01778930053114891 -> 0.012172756716609001`;
- loss ratio: `0.6842740497466221` versus gate `<= 0.95`;
- WD activity delta: `0.007103331430698745`;
- outputs finite: `true`;
- locked test touched: `false`;
- quality claim: none; this is numerical/trainability sanity only.

Decision: `V32_P1_MINI_OVERFIT_OK`.

Audit note: P1 trained over `8` loaded crops, but its initial/final loss gate
and WD activity check were computed only on `inputs[:batch_size]`, i.e. the
first `2` crops. This is enough to show the code could train, but it is not the
correct aggregate P1 sufficiency evidence.

## P1b Aggregate Result

P1b ran on `convir-4090` from route commit `31fbb01`.

- sample contract: `8` train-derived center crops, crop size `256`;
- aggregate metric contract: initial/final loss, finite outputs, and WD
  activity measured over all `8` loaded crops in eval mode, chunked as
  `[(0,2), (2,4), (4,6), (6,8)]`;
- train scope: `wd_decoder`;
- trainable params: `4453433`;
- aggregate initial/final loss:
  `0.01766193099319935 -> 0.013848769944161177`;
- aggregate loss ratio: `0.7841028225902132` versus gate `<= 0.95`;
- WD activity delta: `0.005941152640540774`;
- outputs finite: `true`;
- locked test touched: `false`;
- quality claim: none; this is aggregate numerical/trainability sanity only.

Decision: `V32_P1B_AGGREGATE_MINI_OVERFIT_OK`.

## Closeout

Decision label:
`COMPLETED_P0_P1B_AGGREGATE_PASS_P2_GATE_FAIL_LOCKED_TEST_BLOCKED`.

P0/P1b validate that the route branch builds, partial-loads the official A0
checkpoint, starts as an exact no-op, and can train the declared WD/decoder
scope on a tiny train-derived sanity set without numerical pathology under the
correct all-loaded-crop aggregate gate. P2 then tested the declared fixed
train-derived validation screen and failed the model-line quality gate.

Next allowed action: no P3, no locked test, and no further v3.2 continuation
under the current gate. Future full-model work requires a new written
route/design.

## P2 Design Status

P2 is defined as a fixed train-derived validation screen using the v3.1
600-image raw table for split identity:

- validation: `fold_id=0`, 120 train-derived images;
- training: `fold_id=1..4`, 480 train-derived images;
- scope: `wd_decoder`;
- seed: `3407`;
- budget: 20 epochs, batch size `4`, save/validate every 5 epochs;
- primary checkpoint: `Best.pkl`, selected by validation PSNR only;
- baseline: official A0 on the same 120 validation images;
- locked test: blocked.

The P2 gate and stop/continue rules are written in
`v32_p2_train_derived_validation_design.md`.

Initial P2 launch note: commit `8d7a9f4` failed at epoch 5 in the auxiliary
modulation-stat logging path because full validation images were not padded
before `collect_wd_stats`. This is recorded as
`PREFLIGHT_FAILED_ENGINEERING`, not a scientific quality result. The corrected
P2R1 rerun pads that stats path and preserves the same split, hyperparameters,
checkpoint policy, and gate.

## P2 Result

Corrected P2R1 ran to completion on `convir-4090` from route commit `30077ee`
with run id `ConvIR-Haze4K-v32-p2r1-wddecoder-seed3407-20260707`.

Primary checkpoint `Best.pkl` was selected only by P2 validation PSNR. Against
official A0 on the same 120 validation images, Best produced:

- mean PSNR delta: `+0.13874422709147136`;
- hard-bottom25 PSNR delta: `+0.19586575826009114`;
- easy-top25 PSNR delta: `+0.047612508138020836`;
- p05 PSNR delta: `-0.5714302062988281`;
- CVaR5 PSNR delta: `-0.7218182881673177`;
- mean SSIM delta: `+0.000042928755283355714`;
- catastrophic proxy count: `0`.

Supportive `Final.pkl` also failed: mean/hard/easy PSNR deltas were
`+0.09675443967183431 / +0.08151111602783204 / +0.05684814453125`, with
p05/CVaR5 `-0.47830810546875 / -0.5908279418945312`.

Decision:
`V32_P2_TRAIN_DERIVED_VALIDATION_FAIL_OR_NOT_COMPETITIVE_LOCKED_TEST_BLOCKED`.

The route is scientifically negative under the predeclared P2 gate. P3 and
locked test are not authorized.
