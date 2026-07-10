# CHD-RM v2i FAM2 No-Op Architecture Equivalence

Status: `COMPLETED_GATE_PASS`

Decision label: `V2I_FAM2_NOOP_ARCH_EQUIVALENCE_PASS_AUTHORIZE_D7C_GATED_NOOP_CONNECTION_ONLY`

Route card:
`experience_docx/experiment_cards/haze4k-chd-rm-v2i-fam2-noop-arch-equivalence.md`

Central index: `experience_docx/CHD_RM_EXPERIMENT_INDEX.md`

## Purpose

v2i tests only whether a FAM2 zero-init modulation shell can be inserted from
the official ConvIR-B architecture anchor while preserving exact A0 behavior.

## Authorized Work

- FAM2-only zero-init architecture insertion.
- Strict partial-load compatibility audit.
- Random tensor no-op equivalence.
- Real Haze4K train-derived batch no-op equivalence.
- Internal val-inner 600 metric and output equivalence summary.
- Modulator zero-stat audit.

## Not Authorized

- No locked Haze4K test.
- No RARM connection or training.
- No D7c gate injection into the forward path.
- No adapter training, ConvIR-B unfreeze, loss changes, F5/D2/v3 expansion, or
  canary expansion.
- No weights, checkpoints, images, arrays, archives, or raw inference outputs.

## Primary Files

- `v2i_route_decision.md`
- `v2i_source_of_truth_manifest.json`
- `no_locked_test_audit.json`
- `forbidden_flow_audit.json`
- `coverage_contract.md`
- `fam2_arch_patch_summary.md`
- `fam2_noop_protocol.md`
- `run_v2i_fam2_noop_equivalence.sh`
- `status.txt`

Runtime result files:

- `fam2_state_dict_compatibility.json`
- `fam2_param_delta_audit.json`
- `fam2_noop_random_equivalence.json`
- `fam2_noop_real_batch_equivalence.json`
- `fam2_noop_internal_val600_summary.json`
- `fam2_noop_per_scale_diff_summary.csv`
- `fam2_modulation_zero_stats.json`
- `fam2_noop_closeout.json`
- `v2i_next_stage_decision.md`

## Result

The cloud audit completed successfully on `convir-4090`.

Key metrics:

- candidate missing keys: exactly `FAM2.modulator.weight`,
  `FAM2.modulator.bias`;
- unexpected keys: `[]`;
- shape mismatches: `[]`;
- parameter delta: `8320` (`0.09640045118191935%`);
- random tensor max/mean abs diff: `0.0` / `0.0`;
- real train-derived batch max/mean abs diff: `0.0` / `0.0`;
- internal val-inner 600 samples: `600`;
- internal val-inner 600 max abs diff: `0.0`;
- PSNR/SSIM max absolute deltas: `0.0` / `0.0`;
- locked Haze4K test usage: `none`;
- training/RARM/D7c forward connection: `none`.

Decision: FAM2-only zero-init architecture insertion is A0-equivalent. The only
authorized next experiment is a separate D7c-gated no-op connection audit; RARM
training remains blocked.
