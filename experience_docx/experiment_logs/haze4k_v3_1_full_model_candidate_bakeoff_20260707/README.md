# Haze4K v3.1 Full-Model Candidate Bakeoff

Status: COMPLETED_GATE_PASS_WDMAMBA_FULL_MODEL_BASELINE_HEADROOM.

Purpose: separate standard full-model quality from strict A0-dominance safe-upgrade. This route is diagnostic-only and train-derived only.

Sources: v2.37 WDMamba full-image table, v2.39 ConvIR-L full-image table, and v2.2 FullUDP full-image table joined on the same 600 train-derived image names. Locked test remains untouched.

Forbidden: no canary80, no locked test, no A0+alpha deployable selector, no bridge/generator, no v3.0 rescue.

Key result: WDMamba standalone is the strongest full-model candidate in this bakeoff. Mean delta vs official ConvIR-B A0 is 3.5778 dB; hard/easy deltas are 8.2765/-1.0483 dB. ConvIR-L standalone is also positive at 1.0945 dB mean, while FullUDP standalone is negative at -0.4313 dB mean on this joined table.

Decision: use v3.1 as evidence to pivot away from ConvIR-B A0-anchored rescue and draft v3.2 as a full model line centered on WDMamba/ConvIR-WD-style low-frequency haze modeling.

Compact artifacts:
- `v31_candidate_metric_matrix.csv`
- `v31_candidate_tail_matrix.csv`
- `v31_candidate_oracle_upper_bound.json`
- `v31_candidate_overlap_matrix.csv`
- `v31_candidate_cost_manifest.json`
- `v31_closeout.json`

Cloud-only raw table: `v31_candidate_per_image_cloud_only.csv`.
