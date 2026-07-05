# v2.31 P0 Architecture Contract Delta

branch: `codex/haze4k-v2-31-nopost-action-value-identifiability-audit`
commit: `46122af`
parent_branch: `codex/haze4k-v2-30-nopost-ilfrb-acs-compatibility-gated-oof-table-policy`
parent_commit: `8971902`
checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`
checkpoint_sha256: `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`

v2.31 does not add runtime model structure. It reuses the v2.30
NoPost ILFRB-ACS audit stack and adds only train-derived target-only
action-value identifiability diagnostics.

runtime_forward_contract: `forward(self, x)`
teacher_or_expert_forward_input: `false`
rgb_output_output_residual: `false`
learned_rgb_post_output_correction: `false`
p2b_selector_probe_launched: `false`
training_launched: `false`
locked_test_touched: `false`
forbidden_symbol_hits: `0`
decision: `P0_PASS_ARCH_CONTRACT_DELTA_AUDIT`
