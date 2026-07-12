# v3m-A0a Data Contract

- Source split: Haze4K `train` only.
- OOF panel: `v3j_controller_train`, up to 1200 clean-reference groups.
- Confirm panel: `v3j_route_confirm`, up to 600 images, audit-only.
- Group key: clean-reference name from the v3j split manifest.
- Paired comparison: every policy is evaluated on the same group set per
  operator.
- Locked Haze4K test: not enumerated for metrics and not read by the command.
- Missing or row-order mismatch: engineering failure; do not interpret a gate.
