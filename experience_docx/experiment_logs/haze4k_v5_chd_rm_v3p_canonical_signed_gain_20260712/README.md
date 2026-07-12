# v3p Canonical Signed-Gain Evidence

Status: `A2_COMPLETED_GATE_PASS`

The historical v3o fail-stop remains unchanged. v3p established a new
float64 canonical candidate-loss contract, then performed read-only
reconstruction and oracle measurements; it did not revise v3o evidence.

## Compact Stage Record

- A0 numerical preflight, smoke, and formal OOF passed. The formal closeout
  `v3p_a0_closeout.json` authorizes A1 reconstruction only.
- A1's first bin reader is recorded as an engineering failure. A1r repairs
  only its bin-boundary semantics and passes full action-count reconstruction;
  `v3p_a1r_closeout.json` authorizes A2 only.
- A2 is `COMPLETED_GATE_PASS` in `v3p_a2_closeout.json`:
  `V3P_A2_CONSTRAINED_G1_ORACLE_PASS_AUTHORIZE_B0_PHYSICS_FORWARD_CONTRACT_ONLY`.
  The per-operator LCB95 lift over uniform `.125` is `+0.045132 dB` (`D_ref`)
  and `+0.045011 dB` (`D_rep`); over uniform `.25` it is `+0.021617 dB` and
  `+0.021320 dB`. Neither operator has a severe fixed-baseline regression or
  selected harmful SSE. `v3p_a2_by_fold.csv` contains the 5x2 aggregate
  stability table.

## Evidence Boundaries

`v3p_a2_per_image_cloud_only.csv`, the formal block table, logs, and
`status.txt` remain cloud-only under:

`/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3p_canonical_signed_gain_20260712`

The compact A2 package contains only `v3p_a2_closeout.json`,
`v3p_a2_summary.json`, `v3p_a2_source_manifest.json`, and
`v3p_a2_by_fold.csv`. It authorizes B0 physics-forward contract work only;
it is not a deployed selector, trained controller, policy replay, canary, or
locked-test result.
