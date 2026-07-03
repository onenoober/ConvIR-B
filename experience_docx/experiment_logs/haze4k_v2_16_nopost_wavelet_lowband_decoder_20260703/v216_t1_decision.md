# v2.16 T1 Decision

Decision: `T1_LOWBAND_HEADROOM_PASS_ALLOW_T2`

Gate criteria:

- all-image LL oracle mean dPSNR >= `0.20`: `14.998694`
- A0 hard-bottom25 LL oracle mean dPSNR >= `0.30`: `18.939359`
- A0 easy-top25 LL oracle mean dPSNR >= `-0.05`: `11.853745`
- severe LL-oracle regressions <= `max(3, 1%)`: `0`
- lowband-need rate >= `0.20`: `1.000000`

Locked Haze4K test remains untouched. No training was launched.
