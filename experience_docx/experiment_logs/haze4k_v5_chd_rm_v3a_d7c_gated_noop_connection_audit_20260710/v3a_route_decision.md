# v3a Route Decision

Decision before launch:
`AUTHORIZED_BY_V2I_NO_TRAINING_NO_RARM`

v3a is authorized only as a D7c-gated no-op connection audit. It may pass D7c
gate tensors into FAM2 if the final modulation remains mathematically no-op and
exact A0 equivalence is re-proven.

Still blocked:

- RARM training;
- adapter training;
- ConvIR-B unfreeze;
- loss changes;
- locked Haze4K test;
- canary expansion.

## Final Decision

`V3A_D7C_GATED_NOOP_CONNECTION_PASS_AUTHORIZE_NO_TRAINING_RARM_PREFLIGHT_ONLY`

Cloud attempt 5 passed after fixing an audit-only missing-key order comparison.
The candidate keeps A0 exact equivalence while accepting nontrivial D7c gate
tensors into the FAM2 no-op shell.

Final evidence:

- `d7c_noop_closeout.json`: `pass=true`;
- `d7c_noop_state_dict_compatibility.json`: `param_delta=8320`, expected
  missing keys only, no unexpected keys, no shape mismatches;
- `d7c_noop_internal_val600_summary.json`: `600` train-derived internal
  val-inner samples, output max diff `0.0`, PSNR delta max abs `0.0`, SSIM delta
  max abs `0.0`, nontrivial D7c gate images `599`;
- `forbidden_flow_audit.json`: no locked test, no training, no RARM, no adapter
  training, no ConvIR-B unfreeze.

Still blocked after pass:

- RARM training;
- adapter training;
- ConvIR-B unfreeze;
- loss changes;
- locked Haze4K test;
- canary expansion.

Only a separate preflight/design decision can define the next route.
