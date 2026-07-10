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
