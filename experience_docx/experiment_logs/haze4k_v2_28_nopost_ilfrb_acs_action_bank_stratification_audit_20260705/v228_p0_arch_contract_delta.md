# v2.28 P0 Architecture Contract Delta

branch: `codex/haze4k-v2-28-nopost-ilfrb-acs-action-bank-stratification-audit`
commit: `aa9676e`
parent_branch: `codex/haze4k-v2-27-nopost-ilfrb-action-conditioned-selective-distill`
parent_commit: `de5a68b`
checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`
checkpoint_sha256: `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`

v2.28 does not add a new model structure on top of v2.27. It reuses the
NoPost ILFRB-ACS architecture and replaces only the action-bank replay
diagnostic protocol with out-of-fold prototypes, cross-sample bucket swaps,
and diagnostic negative controls.

runtime_forward_contract: `forward(self, x)`
teacher_or_expert_forward_input: `false`
rgb_output_output_residual: `false`
learned_rgb_post_output_correction: `false`
training_launched: `false`
locked_test_touched: `false`
forbidden_symbol_hits: `0`
decision: `P0_PASS_ARCH_CONTRACT_DELTA_AUDIT`
