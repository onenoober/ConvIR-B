# Haze4K v3.2 ConvIR-WD Full Model Line Evidence

Status: `COMPLETED_P0_P1B_AGGREGATE_GATE_PASS_P2_DESIGN_OPEN_LOCKED_TEST_BLOCKED`.

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
- P2 train-derived validation: design open only; do not launch until the split,
  budget, checkpoint policy, and gate are written.

Locked test policy: locked Haze4K test is blocked.

Primary outputs:
- `v32_p0_preflight.json`;
- `preflight_v32.log`;
- `v32_p1_mini_overfit.json`;
- `p1_mini_overfit_v32.log`;
- `v32_p1b_mini_overfit_aggregate.json`;
- `p1b_mini_overfit_aggregate_v32.log`;
- `run_p1b_mini_overfit_aggregate.sh`;
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
`COMPLETED_P0_P1B_AGGREGATE_GATE_PASS_P2_DESIGN_OPEN_LOCKED_TEST_BLOCKED`.

P0/P1b validate that the route branch builds, partial-loads the official A0
checkpoint, starts as an exact no-op, and can train the declared WD/decoder
scope on a tiny train-derived sanity set without numerical pathology under the
correct all-loaded-crop aggregate gate. They do not prove Haze4K quality or
model-line superiority.

Next allowed action: write a P2 train-derived validation design before any
larger training. Locked test remains blocked.
