# v3o Deviation Log

## 2026-07-12 A0 smoke launcher engineering failure

The first A0 smoke launcher created tmux session `v3o_a0`, then the durable
runner checked whether that same session existed and exited before invoking
Python. No model forward, candidate table, policy replay, training, canary, or
locked-test access occurred. The runner now leaves session-conflict detection to
the external launcher, which performs that check before creating the session.
