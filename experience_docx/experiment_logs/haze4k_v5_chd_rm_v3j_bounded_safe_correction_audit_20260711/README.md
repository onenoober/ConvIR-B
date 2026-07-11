# v3j Bounded Safe-Correction Audit Evidence

Route: `haze4k_v5_chd_rm_v3j_bounded_safe_correction_audit_20260711`
Branch: `codex/haze4k-v5-v3j-bounded-safe-correction-audit`
Status: closed, no promotion

## Source Of Truth

- Parent runnable branch: `codex/haze4k-v5-v3i-fam2-open-value-distillability`
  at commit `8517614`.
- Runtime worktree:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3j-bounded-safe-correction-audit`.
- Runnable route commit: `601763c`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Dataset: Haze4K train-derived internal splits only. Locked test untouched.

## Result

v3j-A passed: the bounded output-residual action space is safe when supervised
by the privileged primary teacher. The strongest deployable projection
`PRIMARY_FULL_CLIP_P99_D7C` reached mean `+0.229641`, p10 `+0.002454`,
worst `-0.018076`, severe `0`, and bootstrap CI95 low vs hard `+0.199968`
on `v3j_route_confirm`.

v3j-B failed: tiny direct residual heads learned positive mean deltas but
created unsafe tails on both OOF and route-confirm. No direct head passed the
dual OOF + confirm gate.

## Bottleneck

The bottleneck is not the bounded residual actuator and not mean-value
learnability. It is deployable tail safety: learned residual predictions
produce many direct-only severe regressions even while average PSNR improves.

Key route-confirm numbers:

- hard D7c: mean `+0.008634`, p10 `-0.131343`, severe `23`.
- direct linear: mean `+0.057145`, p10 `-0.392219`, severe `121`.
- direct context: mean `+0.099199`, p10 `-0.444083`, severe `121`.

## Decision

`V3J_DIRECT_SAFE_CORRECTION_OOF_FAIL_REQUIRE_NEW_INFORMATION_NO_INTERNAL_ROUTER`

No v3j no-op architecture equivalence, canary, backbone/FAM training, or
internal-router continuation is authorized from this evidence. A future route
must add genuinely new information for tail-risk control before any model
promotion attempt.

## Compact Evidence

- `v3j_a_bounded_action_audit_summary.json`
- `bounded_action_space_replay_summary.csv`
- `bounded_action_space_bootstrap.csv`
- `bounded_action_space_bounds.json`
- `fresh_route_confirm_split_manifest.json`
- `direct_correction_probe_summary.json`
- `direct_correction_oof_policy_summary.csv`
- `direct_correction_oof_bootstrap_vs_hard.csv`
- `direct_correction_route_confirm_policy_summary.csv`
- `direct_correction_route_confirm_bootstrap_vs_hard.csv`
- `direct_correction_tail_audit.json`
- `direct_correction_probe_training_history.csv`
- `status.txt`

Raw per-image replay tables and logs remain on cloud only by default.
