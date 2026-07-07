# Haze4K v4.6 DCFSB-Bottleneck Evidence README

Route id: `haze4k_v4_6_dcfsb_bottleneck_20260708`

Branch: `codex/haze4k-v4-6-dcfsb-bottleneck-independent`

Runtime host: `convir-4090`

Cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Policy: train on `adapter_train`, audit on `internal_holdout256`, locked test blocked.

## Preflight Gate

Status: passed; adapter-5 training authorized.

- Branch/commit: `codex/haze4k-v4-6-dcfsb-bottleneck-independent` / `3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`
- Checkpoint SHA256: `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`
- Split contract: adapter_train `2744`, internal_holdout256 `256`, overlap `0`
- Init equivalence to A0: synthetic max abs `0.0`, train-crop max abs `0.0`
- Added params: `29732`
- Trainable prefixes: `['DCFSB_Bottleneck']`
- Missing partial-load keys are restricted to new DCFSB module: `18`
- Locked test touched/enumerated: `False` / `False`

Next authorized phase: run `run_v4_6_dcfsb_adapter5_notest.sh` with `--valid_freq 999`, then audit trainfit128 and internal_holdout256 with high-frequency L1 tracking.

## Rescue Probe Authorization

The adapter5 audit is scientifically promising but fails the stage gate only on p5 tail risk. To avoid prematurely rejecting a potentially useful route, two train-derived rescue probes are authorized before closeout:

- `adapter3`: same LR, 3 epochs, testing whether lower adapter strength preserves the mean while reducing p5 harm.
- `adapter5_lr5e5`: 5 epochs at half LR, testing whether slower adaptation reduces tail harm.

Both probes keep `--valid_freq 999`, train only on `adapter_train`, audit only trainfit128/internal_holdout256, and keep locked test blocked.

## Adapter4 Bracket Probe Authorization

The first rescue probes bracket the trade-off: adapter5 passes mean dPSNR but fails p5, while adapter3 passes p5 but narrowly misses mean dPSNR. A final adapter4 probe is authorized to test the midpoint. It keeps the same locked-test policy, training split, seed, and metric contract.

## Audit Closeout

Status: internal stage gate passed.

Selected candidate: `adapter4`.

- adapter4 internal256 mean dPSNR `0.044404`, positive ratio `0.625000`, p5 dPSNR `-0.216141`
- adapter4 high-frequency L1 delta `-0.00000057`, high-gate std `0.038074`
- adapter5 was mean-positive but tail-risky; adapter3 was tail-safe but just under mean threshold; low-LR was too weak.
- Locked test touched/enumerated: `False` / `False`
- Raw per-image/module tables, checkpoints, and TensorBoard runs remain cloud-only.

Decision: keep v4.6 DCFSB-bottleneck adapter4 as the current candidate. Any locked-test or broader validation needs a separate written gate.
