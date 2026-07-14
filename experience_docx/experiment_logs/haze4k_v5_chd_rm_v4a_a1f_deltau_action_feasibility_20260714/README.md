# v4a-A1F Delta-u Action Feasibility Evidence

Status: `COMPLETED_GATE_PASS`; A1F formal only is authorized.

This route is a privileged, train-derived feasibility oracle after v4a A0P
found no local optimizer/projection/window correction. It restores the exact
A0R r1 final v3z head and tests whether a bounded direction-line action set
adds safe heldout128 headroom beyond privileged shrink/abstention.

The canonical contract is:

`experience_docx/experiment_cards/2026-07-14-haze4k-v5-v4a-a1f-deltau-action-feasibility.md`.

No training, policy fitting, canary, candidate selection, or locked-test access
is permitted. Raw per-image/action rows remain under cloud `RUN_ROOT`; only the
source manifest, typed closeout, operator summary, bootstrap summary, and this
README may be compact GitHub evidence.

The first smoke attempt, `v4a_a1f_s0_smoke_r1` at route commit `ff9cf921`,
stopped before a gate decision after 8 update and 3 heldout images. The fixed
shrink grid re-reduced its guaranteed zero action in a batched float32 kernel;
the `1e-12` comparison could then exclude the exact predecessor. The repair
requires bitwise equality of the zero-action rendered tensors and canonicalizes
only that identical candidate to the already replayed predecessor metrics.
Other candidates and all scientific thresholds are unchanged. See
`v4a_a1f_smoke_r1_failure_closeout.json`.

The repaired smoke, `v4a_a1f_s0_smoke_r2` at route commit `42dbbf18`, passed
the frozen S0 contract:

- typed tuple: `COMPLETED_GATE_PASS` /
  `V4A_A1F_S0_ALIGNMENT_PASS_AUTHORIZE_FORMAL_ONLY` / `A1F_FORMAL_ONLY`;
- 8 update plus 8 heldout images, both operators, 32 complete rows;
- exact A0D replay: maximum MSE and PSNR discrepancies `0.0`;
- zero-action tensor, bound, and support excess maxima `0.0`;
- batched zero-grid MSE drift, retained only as an engineering diagnostic:
  `1.1641532182693481e-10`;
- severe and hard regressions versus old `.25`: `0` in every split/operator;
- runtime `4.0465 s`, peak allocated GPU memory `618.27 MiB`;
- training, candidate selection, canary, and locked-test access all `false`.

Raw rows and the full log remain under
`/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v4a_a1f_deltau_action_feasibility_20260714/v4a_a1f_s0_smoke_r2/`
and the route `RUN_ROOT`; they are not Git evidence. The compact smoke package
is the closeout, source manifest, operator summary, bootstrap-not-run marker,
failure closeout, and this README.
