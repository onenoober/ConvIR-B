# v2.18 P1 O1 Action Learnability Decision

Decision: `P1_FAIL_O1_GLOBAL_ACTION_NOT_SAFELY_LEARNED_BY_POOLED_LL_POLICY`

Primary replay summary:

- mean dPSNR: `0.26317835489908853`
- hard bottom25 dPSNR: `0.8594181060791015`
- easy top25 dPSNR: `-0.18392913182576498`
- p05 dPSNR: `-1.164642333984375`
- CVaR5 dPSNR: `-2.0502514680226644`
- severe rate: `0.23666666666666666`
- strong-reference regressions: `303` / `600`
- control gap vs shuffled: `0.308957724571228`

Interpretation:

- P1 is about deployable policy learnability, not oracle headroom.
- If P1 fails while v2.17 O2/O3 remains strong, WLDB-A2 global policy should not train yet; design spatial WLDB-B learnability next.
- Locked Haze4K remains untouched.
