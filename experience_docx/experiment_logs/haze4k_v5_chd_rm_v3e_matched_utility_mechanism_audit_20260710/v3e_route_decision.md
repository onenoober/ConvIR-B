# v3e Route Decision

Date: 2026-07-10

Decision:
`V3E_AUTHORIZE_MECHANISM_AUDIT_NO_TRAINING_NO_LOCKED_TEST`

The v3d route remains paused. v3e is authorized only to inspect the existing
v3d D7c-gated and ungated-control artifacts and to run no-training mechanism
audits on the internal val-inner 600 split.

Authorized:

- paired bootstrap/sign/tail reanalysis from existing v3d CSV;
- no-training replay of existing weights under D7c hard gate and all-ones gate;
- operator-gain alignment and boundary leakage audit;
- no-step gradient and training-contract audit.

Not authorized:

- any training continuation;
- any new checkpoint-producing experiment;
- any loss/optimizer/scheduler behavior change;
- any locked-test access.
