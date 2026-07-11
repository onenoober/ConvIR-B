# v3k Closeout

Decision: `V3K_PROVISIONAL_MICRO_ALPHA_SAFE_STEP_SUPPORTED_NO_CANARY_NO_NEW_SEALED_SPLIT`.

Strict reconstruction failed: row counts and identities matched v3j-B, but direct-head PSNR deltas and severe sets differed. Therefore v3k diagnostic outputs are labeled as new replicate/provisional evidence, not authorization evidence.

Bottleneck: A0-relative correction advantage sign and safe-step observability. Full direct steps mix wrong-direction samples with harmful overshoot; alpha* uses the critical harmful full-step threshold 0.5.

Key direction counts:
- OOF_DIRECT_CONTEXT: wrong_direction=328, harmful_overshoot=264, alpha_p50=0.519, severe_le_0p2=279.
- OOF_DIRECT_LINEAR: wrong_direction=342, harmful_overshoot=326, alpha_p50=0.344, severe_le_0p2=233.
- CONFIRM_DIRECT_CONTEXT: wrong_direction=155, harmful_overshoot=135, alpha_p50=0.552, severe_le_0p2=125.
- CONFIRM_DIRECT_LINEAR: wrong_direction=170, harmful_overshoot=145, alpha_p50=0.458, severe_le_0p2=116.

OOF fixed-alpha diagnostic:
- OOF_FIXED_ALPHA_0.125_CONTEXT: mean=0.0298, p10=-0.0279, severe_le_0p2=0, le_0p5=0, paired_vs_hard_mean=0.0129, CI95=[0.0067,0.0195].
- OOF_FIXED_ALPHA_0.25_CONTEXT: mean=0.0536, p10=-0.0636, severe_le_0p2=15, le_0p5=0, paired_vs_hard_mean=0.0366, CI95=[0.0290,0.0444].
- OOF_HARD_D7C_ALPHA1: mean=0.0169, p10=-0.1061, severe_le_0p2=35, le_0p5=0.

Open val_inner holdout, not sealed:
- seed3407 v3k_val_inner_open_holdout_FIXED_ALPHA_0.125_CONTEXT: mean=0.0287, p10=-0.0324, severe_le_0p2=0, le_0p5=0, paired_vs_hard_mean=0.0159, CI95=[0.0062,0.0255].
- seed3407 v3k_val_inner_open_holdout_FIXED_ALPHA_0.25_CONTEXT: mean=0.0512, p10=-0.0745, severe_le_0p2=10, le_0p5=0, paired_vs_hard_mean=0.0384, CI95=[0.0275,0.0493].
- seed3407 v3k_val_inner_open_holdout_HARD_D7C_ALPHA1: mean=0.0128, p10=-0.1215, severe_le_0p2=23, le_0p5=0.

Seed stability on open val_inner:
- seed3408 v3k_val_inner_open_holdout_FIXED_ALPHA_0.125_CONTEXT: mean=0.0287, p10=-0.0315, severe_le_0p2=0, paired_vs_hard_CI95=[0.0063,0.0250].
- seed3408 v3k_val_inner_open_holdout_FIXED_ALPHA_0.25_CONTEXT: mean=0.0514, p10=-0.0717, severe_le_0p2=9, paired_vs_hard_CI95=[0.0279,0.0492].

No canary is authorized. Next real promotion step requires a genuinely new sealed train-derived or external validation split plus deterministic saved head artifacts.
