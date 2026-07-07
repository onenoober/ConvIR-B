# v3.0 A0-Anchored Partial-Unfreeze Risk-Controlled ConvIR

Purpose: test whether moving beyond frozen ConvIR-B small residual heads can produce a deployable, non-post, tail-safe improvement over A0.

Starting point: official ConvIR-B architecture anchor and checkpoint.

Architecture:
- initialize from official ConvIR-B;
- zero-init any new low-frequency/global-context branch;
- allow partial unfreeze of final decoder / reconstruction head;
- no post-processing;
- Stage-0 output must match A0.

Primary losses:
- GT reconstruction loss;
- A0 preservation loss on easy/strong-reference train buckets;
- hinge loss vs A0 MSE;
- CVaR/top-k relative-MSE loss;
- residual direction loss: `max(0, <A0-GT, Y-A0>)`;
- optional teacher auxiliary only on offline GT-safe subset.

Stage-0 gates:
- strict/partial load clean;
- identity max_abs_vs_A0 <= 1e-7;
- forbidden symbol hits = 0;
- locked test untouched;
- trainable parameter manifest synced as compact text only.

Canary32 OOF gates:
- mean_delta >= +0.15 dB;
- hard_delta >= +0.30 dB;
- easy_delta >= +0.00 dB;
- p05 >= -0.01 dB;
- CVaR5 >= -0.02 dB;
- severe = 0;
- strong-reference regressions = 0;
- fold_pass >= 4/5;
- severe_direction_bad = 0.

Promotion policy: canary80 is blocked unless canary32 OOF passes. Locked test is blocked unless canary80 fixed-selection passes.
