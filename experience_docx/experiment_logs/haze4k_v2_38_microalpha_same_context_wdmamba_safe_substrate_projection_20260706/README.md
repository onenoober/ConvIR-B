# Haze4K v2.38 Micro-Alpha Same-Context WDMamba Evidence

Status: `COMPLETED_P0_GATE_FAIL`

Route card:
`experience_docx/experiment_cards/2026-07-06-haze4k-v2-38-microalpha-same-context-wdmamba-safe-substrate-projection.md`

Central index path:
`experience_docx/EXPERIMENT_INDEX.md`

Runtime host: `convir-4090`

Cloud workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-38-microalpha-same-context-wdmamba-safe-substrate-projection`

Cloud Python:
`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Locked-test policy: blocked for all phases in this route.

## Evidence Files

Compact sync candidates:

- `status.txt`
- `run_v238_p0_p2_microalpha_audits.sh`
- `v238_p0_microalpha_safety_sweep_summary.json`
- `v238_p0_closeout.json`
- `v238_decision_tree.md`
- `v238_closeout.json`

Cloud-only runtime/raw evidence:

- `v238_p0_microalpha_safety_sweep_per_image.csv`

P1/P2/P3 files were intentionally not generated because P0 found no passing
micro-alpha.

## Metric Contract

P0 is an offline audit over the v2.35/v2.37 full-image same-context cache. It
does not rerun WDMamba, train bridge/generator models, use canary80, or touch
locked test.

Gate summary: image_count `600`, cache_sha_coverage
`1.0`, strict no-selector full600/fold gate.

| alpha | mean | hard | easy | p05 | CVaR5 | worst | severe | strong-reg | fold pass | gate pass |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0.015625 | +0.1073 | +0.1262 | +0.0806 | +0.0492 | +0.0325 | -0.0267 | 0 | 0 | 0/5 | False |
| 0.03125 | +0.2148 | +0.2539 | +0.1594 | +0.0963 | +0.0625 | -0.0549 | 0 | 1 | 0/5 | False |
| 0.046875 | +0.3224 | +0.3831 | +0.2362 | +0.1392 | +0.0897 | -0.0845 | 0 | 1 | 0/5 | False |
| 0.0625 | +0.4301 | +0.5140 | +0.3110 | +0.1827 | +0.1139 | -0.1156 | 0 | 1 | 3/5 | False |
| 0.078125 | +0.5378 | +0.6465 | +0.3835 | +0.2224 | +0.1351 | -0.1481 | 0 | 2 | 2/5 | False |
| 0.09375 | +0.6454 | +0.7807 | +0.4538 | +0.2539 | +0.1534 | -0.1820 | 0 | 2 | 2/5 | False |
| 0.109375 | +0.7530 | +0.9165 | +0.5216 | +0.2854 | +0.1683 | -0.2172 | 0 | 3 | 2/5 | False |
| 0.125 | +0.8605 | +1.0540 | +0.5869 | +0.3100 | +0.1798 | -0.2537 | 0 | 3 | 2/5 | False |

## Result

Decision: `P0_FAIL_STOP_NO_MICROALPHA_SAFE_SUBSTRATE`.

No micro-alpha passed the strict no-selector gate. The smallest alpha preserved
tail safety best but had too little mean/hard gain; larger alphas gained more
but reintroduced worst-delta, strong-reference, and fold-stability failures.
P1 OOF alpha selection, P2 safety-margin audit, P3 free-tensor projection,
bridge/generator training, canary80, and locked test remain blocked.
