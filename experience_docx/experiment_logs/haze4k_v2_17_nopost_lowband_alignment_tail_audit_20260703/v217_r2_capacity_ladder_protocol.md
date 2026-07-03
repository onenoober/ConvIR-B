# v2.17 R2 Capacity Ladder Protocol

No deployable model is trained here. Each oracle optimizes per-image bounded feature-LL corrections against train-derived GT only to test representational headroom.

- O0: official A0 identity.
- O1: final-feature LL global per-channel offset.
- O2: final-feature LL bounded spatial correction.
- O3: insertion-point oracles at final, mid, and mid+final feature LL.
- O4: RGB LL oracle reference copied from v2.16 T1.

Correction bound: tanh(raw) times `0.5` channel-wise LL std.
Locked Haze4K test remains untouched.
