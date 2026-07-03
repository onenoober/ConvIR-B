# v2.18 P2 Tail-Aware Objective Replay Decision

Decision: `P2_PASS_TAIL_PRESERVE_REPLAY_COVERS_WLDB_A_FAILURE`

- model_5 severe coverage by tail hinge: `1.0`
- model_5 positive tail-hinge activation rate: `0.0`
- model_5 strong/easy regression coverage by preserve hinge: `1.0`
- model_5 positive preserve-hinge activation rate: `0.0`

Interpretation:

- This is an objective replay, not training.
- A pass means the proposed terms would notice the known v2.16 failure mode; it does not prove a model can optimize them.
- Locked Haze4K remains untouched.
