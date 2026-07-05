# v2.30 P0 Architecture Contract Delta

branch: `codex/haze4k-v2-30-nopost-ilfrb-acs-compatibility-gated-oof-table-policy`
commit: `c794602`
parent_branch: `codex/haze4k-v2-29-nopost-ilfrb-acs-safe-oof-action-bank-calibration`
parent_commit: `936e3e0`
checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`
checkpoint_sha256: `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`

v2.30 does not add a new runtime model structure. It reuses the v2.29
NoPost ILFRB-ACS snapshot and changes only the train-derived P2A
compatibility-gated table-policy audit.

runtime_forward_contract: `forward(self, x)`
teacher_or_expert_forward_input: `false`
rgb_output_output_residual: `false`
learned_rgb_post_output_correction: `false`
p2b_selector_probe_launched: `false`
training_launched: `false`
locked_test_touched: `false`
forbidden_symbol_hits: `0`
decision: `P0_PASS_ARCH_CONTRACT_DELTA_AUDIT`
