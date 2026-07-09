# CHD-RM v2f F4b Tail-Rescue Evidence

Status: `COMPLETED_GATE_FAIL`

Route card:
`experience_docx/experiment_cards/haze4k-chd-rm-v2f-need-target-head-redesign.md`

Central index:
`experience_docx/CHD_RM_EXPERIMENT_INDEX.md`

This directory archives the supplemental F4b tail-rescue matrix run on
`convir-4090`. It was launched only after F4 failed, to check whether the F4
failure was caused by insufficient tail pressure rather than a real
safety/recall incompatibility.

Primary files:

- `v2f_f4b_tail_rescue_closeout.json`
- `v2f_f4b_tail_rescue_summary.md`
- `f4b_tail_rescue_matrix_summary.csv`
- `f4b_authorization_record.md`
- `status.txt`
- per-spec text evidence under `tail2_topk10/`, `tail3_cap128_temp04/`, and
  `tail4_topk20/`

Key result:

- `selected_gate_pass_any_variant=false`
- `safe_and_ldhn_point_any_variant=false`
- best safe LDHN recall `0.05233281880197182`
- best selected LDHN recall `0.6660676374862502`, but with unsafe false-tail
- minimum first LDHN-passing false-p95 `0.2`

Decision:

Keep v2f paused. Do not run F5, v3, RARM, D2, or locked Haze4K test from this
evidence state.
