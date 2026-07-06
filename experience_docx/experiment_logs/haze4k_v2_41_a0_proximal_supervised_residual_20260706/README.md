# Haze4K v2.41 A0-Proximal Supervised Residual Evidence

Status: `PLANNED`

Route card:
`experience_docx/experiment_cards/2026-07-06-haze4k-v2-41-a0-proximal-supervised-residual.md`

Central index path:
`experience_docx/EXPERIMENT_INDEX.md`

Runtime host: `convir-4090`

Cloud workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-41-a0-proximal-supervised-residual`

Cloud Python:
`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Locked-test policy: blocked.

## Evidence Files

Compact sync candidates:

- `status.txt`
- `run_v241_p0_stage0_preflight.sh`
- `v241_p0_stage0_preflight.json`
- `v241_p0_closeout.json`

## Metric Contract

P0 Stage-0 proves strict official-checkpoint partial load, zero-init/no-op
equivalence, finite forward, no forbidden postprocess symbols, and locked-test
blocked status. It does not train, run canary80, or touch locked test.

## Result

Pending P0 completion.
