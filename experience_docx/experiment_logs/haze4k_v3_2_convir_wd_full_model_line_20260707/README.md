# Haze4K v3.2 ConvIR-WD Full Model Line Evidence

Status: `PLANNED_P0_P1_ONLY`.

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
- P0 architecture/preflight.
- P1 mini-overfit sanity if P0 passes.

Locked test policy: locked Haze4K test is blocked.

Expected outputs:
- `v32_p0_preflight.json`;
- `preflight_v32.log`;
- `v32_p1_mini_overfit.json`;
- `p1_mini_overfit_v32.log`;
- `status.txt`.
