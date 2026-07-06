# Haze4K v2.39 ConvIR-L Same-Family Teacher Evidence

Status: `COMPLETED_P0_GATE_FAIL`

Route card:
`experience_docx/experiment_cards/2026-07-06-haze4k-v2-39-convirl-same-family-teacher-contract-audit.md`

Central index path:
`experience_docx/EXPERIMENT_INDEX.md`

Runtime host: `convir-4090`

Cloud workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-39-convirl-same-family-teacher-contract-audit`

Cloud Python:
`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Locked-test policy: blocked.

## Evidence Files

Compact sync candidates:

- `status.txt`
- `run_v239_p0_convirl_teacher_sweep.sh`
- `v239_p0_convirl_fullimage_teacher_sweep_summary.json`
- `v239_p0_closeout.json`
- `v239_closeout.json`

Cloud-only runtime/raw evidence:

- `v239_p0_convirl_fullimage_teacher_sweep_per_image.csv`
- ConvIR-L output tensor cache under `/sda/home/wangyuxin/ConvIR-B/runtime_outputs/`

## Metric Contract

P0 runs ConvIR-L full-image inference on the same 600 train-derived images used
by v2.37/v2.38, compares full-image ConvIR-L and A0 in the same context, and
sweeps a strict no-selector alpha grid. It does not train bridge/generator
models, use canary80, or touch locked test.

Gate summary: image_count `600`, cache_sha_coverage
`1.0`, strict no-selector full600/fold gate.

| alpha | mean | hard | easy | p05 | CVaR5 | worst | severe | strong-reg | fold pass | gate pass |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0.015625 | +0.0396 | +0.0282 | +0.0508 | -0.0115 | -0.0262 | -0.0668 | 0 | 0 | 0/5 | False |
| 0.03125 | +0.0789 | +0.0564 | +0.1008 | -0.0231 | -0.0531 | -0.1336 | 0 | 2 | 0/5 | False |
| 0.0625 | +0.1561 | +0.1125 | +0.1984 | -0.0472 | -0.1089 | -0.2675 | 0 | 2 | 0/5 | False |
| 0.125 | +0.3051 | +0.2237 | +0.3823 | -0.1027 | -0.2282 | -0.5358 | 6 | 6 | 0/5 | False |
| 0.25 | +0.5772 | +0.4411 | +0.6990 | -0.2612 | -0.5053 | -1.0704 | 25 | 9 | 0/5 | False |
| 0.5 | +0.9889 | +0.8470 | +1.0757 | -0.6504 | -1.2517 | -2.6008 | 62 | 16 | 0/5 | False |
| 0.75 | +1.1726 | +1.1907 | +1.0422 | -1.2466 | -2.1233 | -5.0680 | 89 | 27 | 0/5 | False |
| 1.0 | +1.0945 | +1.4339 | +0.6015 | -1.9132 | -3.0931 | -7.2091 | 132 | 38 | 0/5 | False |

## Result

Decision: `P0_FAIL_CONVIRL_NO_SAFE_TEACHER_ALPHA`.

No ConvIR-L alpha passed the strict teacher-contract gate. Low alpha values were
tail-safer but too weak, and higher alpha values improved mean/hard metrics only
by introducing unacceptable p05/CVaR5/worst, severe, strong-reference, and fold
failures. P1 free-tensor projection, bridge/generator training, canary80, and
locked test remain blocked.
