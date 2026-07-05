# v2.29 P0 Architecture Contract Delta

branch: `codex/haze4k-v2-29-nopost-ilfrb-acs-safe-oof-action-bank-calibration`
commit: `aa1bd1b`
parent_branch: `codex/haze4k-v2-28-nopost-ilfrb-acs-action-bank-stratification-audit`
parent_commit: `82f4752`
checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`
checkpoint_sha256: `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`

v2.29 does not add a new runtime model structure. It reuses the v2.27/v2.28
NoPost ILFRB-ACS snapshot and changes only the train-derived action-bank
safety-envelope replay protocol.

runtime_forward_contract: `forward(self, x)`
teacher_or_expert_forward_input: `false`
rgb_output_output_residual: `false`
learned_rgb_post_output_correction: `false`
training_launched: `false`
locked_test_touched: `false`
forbidden_symbol_hits: `0`
decision: `P0_PASS_ARCH_CONTRACT_DELTA_AUDIT`
