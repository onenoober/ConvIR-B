# v3i FAM2 Open-Value Distillability Evidence

Route card:
`experience_docx/experiment_cards/haze4k-chd-rm-v3i-fam2-open-value-distillability.md`
Central index: `experience_docx/EXPERIMENT_INDEX.md`

Status: completed on 2026-07-11. Final decision:
`V3I_ALL_DEPLOYABLE_SIGNALS_FAIL_STOP_FAM2_ROUTER_REDESIGN_CANDIDATE`.

v3i was a no-training diagnostic route after v3g/v3h. It used only internal
train-derived `val_inner` 600 evidence, produced no checkpoints, and did not
touch the locked Haze4K test.

## Stage Results

| Stage | Question | Key result | Decision |
| --- | --- | --- | --- |
| v3i-A teacher compressibility | Is the privileged open-value target strong and spatially compressible? | Hard D7c mean `+0.012784 dB` with `23` `<= -0.2 dB` regressions; GT open top50 oracle mean `+0.411695 dB` with `0` regressions; best compressed `ALPHA_SECANT_Q3` mean `+0.412879 dB` with `0` regressions. | `V3I_A_ORACLE_COMPRESSIBLE_AUTHORIZE_FULL_CONTEXT_OOF_PROBE_ONLY` |
| v3i-B full-context OOF probe | Can single-forward/full-context deployable features recover the open-value target? | Best OOF replay `OOF_DW3X3_TOP50` mean `+0.016627 dB`; bootstrap delta vs hard mean `+0.003843`, CI low `-0.000163`. No minimum or strong pass. | `V3I_B_SINGLE_FORWARD_OBSERVABILITY_FAIL_AUTHORIZE_COUNTERFACTUAL_FEATURE_AUDIT_ONLY` |
| v3i-C counterfactual response OOF probe | Does RGB response to alpha counterfactual expose a useful deployable signal? | Best OOF replay `OOF_CF_RESPONSE_DW3X3_TOP50` mean `+0.008543 dB`; delta vs hard mean `-0.004241`, CI low `-0.009492`. No minimum or strong pass. | `V3I_C_COUNTERFACTUAL_RESPONSE_FAIL_AUTHORIZE_DISAGREEMENT_AUDIT_ONLY` |
| v3i-D checkpoint/transform disagreement OOF probe | Does uncertainty/disagreement provide the missing controller signal? | Best OOF replay `OOF_DISAGREE_LINEAR_TOP50` mean `+0.010534 dB`; delta vs hard mean `-0.002250`, CI low `-0.007422`. No minimum or strong pass. | `V3I_ALL_DEPLOYABLE_SIGNALS_FAIL_STOP_FAM2_ROUTER_REDESIGN_CANDIDATE` |

## Bottleneck Assessment

The bottleneck is not FAM2 actuator realizability, not the label-derived
open-value oracle, and not spatial compressibility of the privileged target.
v3i-A shows the open-value oracle is strong and compactly replayable.

The bottleneck is the deployable inference-time controller signal: full-context
single-forward features, alpha counterfactual RGB response features, and
checkpoint/transform disagreement features all failed OOF replay gates. Stop the
current FAM2 router/distillation route. A future route needs materially new
information, target semantics, or a bounded expert / joint correction-confidence
design, and must pass a no-training held-out separability/replay gate before any
training.

## Compact Evidence Files

- `v3i_final_closeout.json`
- `v3i_route_decision.md`
- `no_locked_test_audit.json`
- `v3i_forbidden_flow_audit.json`
- `v3i_a_teacher_compressibility_summary.json`
- `v3i_a_policy_summary.csv`
- `v3i_b_full_context_probe_summary.json`
- `v3i_b_policy_replay_summary.csv`
- `v3i_b_bootstrap_vs_hard.csv`
- `v3i_b_full_tensor_feature_manifest.json`
- `v3i_b_probe_training_history.csv`
- `v3i_c_counterfactual_response_summary.json`
- `v3i_c_policy_replay_summary.csv`
- `v3i_c_bootstrap_vs_hard.csv`
- `v3i_c_counterfactual_feature_manifest.json`
- `v3i_c_probe_training_history.csv`
- `v3i_d_disagreement_summary.json`
- `v3i_d_policy_replay_summary.csv`
- `v3i_d_bootstrap_vs_hard.csv`
- `v3i_d_disagreement_feature_manifest.json`
- `v3i_d_probe_training_history.csv`
- `run_v3i_a_teacher_compressibility_audit.sh`
- `run_v3i_b_full_context_probe.sh`
- `run_v3i_c_counterfactual_response_probe.sh`
- `run_v3i_d_disagreement_probe.sh`
- `status.txt`

## Cloud-Only Raw Outputs

The following files are intentionally left out of GitHub main sync unless a
later audit explicitly needs them: per-image replay CSVs, target/stat tables,
alpha spatial tables, and `.log` files.
