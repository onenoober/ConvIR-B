# v2.33 Failure Root Cause Analysis

Final bottleneck: teacher source exists and is maskable at the table level, but
the current S5-only frozen-ConvIR-B BILFCF carrier compressed
WDMamba-alpha0.5 teacher benefit into near-zero RGB utility on canary32.

Ruled out:

- no teacher source: ruled out by P1;
- gross sign/scale bug: ruled out by P2;
- S5 as worst amplification point: ruled out by P3;
- tail explosion in P4: ruled out by severe `0` and strong-reference regression `0`.

Not ruled out after v2.33:

- exact P4 canary mask coverage mismatch;
- carrier representability gap;
- frozen decoder feature-manifold gap;
- LL-delta objective to RGB PSNR misalignment;
- preservation gradient suppressing teacher-positive signal.

Do not continue:

- no canary80;
- no locked test;
- no more S5-BILFCF steps, samples, or simple loss/mask tuning from this route.
