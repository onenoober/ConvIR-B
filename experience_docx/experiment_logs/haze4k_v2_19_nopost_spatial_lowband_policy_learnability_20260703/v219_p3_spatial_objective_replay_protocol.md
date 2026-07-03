# v2.19 P3 Spatial Objective Replay Protocol

No training is run here. This replay checks whether tail, preserve, and action-budget terms would activate on spatial predictor failures.

Tail hinge audits dPSNR below `-0.15`; severe coverage audits dPSNR below `-0.20`.
Preserve hinge audits strong/easy samples below `-0.05`.
Action budget thresholds are calibrated from O2 oracle and predicted action RMS distributions.
Locked Haze4K remains untouched.
