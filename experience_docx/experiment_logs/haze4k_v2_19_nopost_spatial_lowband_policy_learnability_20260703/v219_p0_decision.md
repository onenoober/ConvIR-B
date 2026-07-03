# v2.19 P0 Contract And Identity Decision

Decision: `P0_PASS_SPATIAL_CONTRACT_IDENTITY_SOURCE_CLEAN`

- contract pass: `True`
- identity rows: `[{'policy_mode': 'spatial', 'image_count': 8, 'max_abs_vs_A0': 0.0, 'mean_abs_vs_A0': 0.0, 'identity_pass_1e-6': True}]`
- param group rows: `[{'policy_mode': 'spatial', 'total_params': 8650217, 'new_nopost_lowband_policy_params': 19552, 'official_params_loaded_or_reused': 8630665, 'missing_key_count': 6, 'unexpected_key_count': 0, 'missing_keys_all_new_prefix': True, 'unexpected_keys': [], 'missing_keys': ['nopost_lowband_policy.policy.context.0.weight', 'nopost_lowband_policy.policy.context.0.bias', 'nopost_lowband_policy.policy.context.2.weight', 'nopost_lowband_policy.policy.context.2.bias', 'nopost_lowband_policy.policy.project.weight', 'nopost_lowband_policy.policy.project.bias']}]`

No training or locked-test command is launched by P0.
