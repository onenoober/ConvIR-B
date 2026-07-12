# v3o Deviation Log

## 2026-07-12 A0 smoke launcher engineering failure

The first A0 smoke launcher created tmux session `v3o_a0`, then the durable
runner checked whether that same session existed and exited before invoking
Python. No model forward, candidate table, policy replay, training, canary, or
locked-test access occurred. The runner now leaves session-conflict detection to
the external launcher, which performs that check before creating the session.

## 2026-07-12 A0 smoke input-contract engineering failure

The corrected launcher invoked the A0 Python entrypoint, which stopped during
input-hash verification before data loading or model forward. The v3o runner
spelled the frozen density artifact hash as
`1ffce13dccb41d96a47c2b5275f87bf2fdbf73c226a190cfa240e5c71c1ec326f`.
The cloud artifact, v3m source manifest, and v3m durable runner agree on
`1ffce13dccb41d96a47c2b5275f87bf2fdb73c226a190cfa240e5c71c1ec326f`.
The runner is corrected to that already-established hash. No candidate output,
policy replay, training, canary, or locked-test access occurred in this failed
attempt.
