# v3m A1 Smoke r2 Closeout

Date: 2026-07-12

Decision:
`V3M_A1_SMOKE_REPLAY_PASS_AUTHORIZE_FORMAL_OOF_ONLY`.

The corrected A1 engineering screen used the exact full 1,200-image OOF fold
map and then evaluated the first 32 manifest names for both frozen operators.
All nine pinned input hashes matched. `D_ref` and `D_rep` both had fixed
`alpha=0.125` maximum absolute PSNR-delta replay difference `0 dB`.

The cloud-only block table contains `40,000` records (`40,001` lines including
header). The compact smoke replay, signal, summary, and source-manifest files
have SHA256 values `a26c40770e47a4330276e1982801f32dccb8d41c58acd875a06483429099b518`,
`868d0d07c9986f5df989b06cd7a4d86e5578d4d2cde7d331a8ccb16cbfd8f54a`,
`cee3f2faca57acc758a9f81d8d0443218aad390d296ac38c30b0662862b9951f`, and
`fc59c7d798c592226eaeb0567b3d9228e3c95e6557bc0ca8a77f3e23b1eaf6d8`.

No smoke signal metric selected a feature or policy. The only authorization is
the formal 1,200-image OOF A1 observability audit with the already frozen
target, signals, directions, and gate. Training, controller calibration,
route-confirm selection, canary, physics/proxy work, and locked-test access
remain blocked.
