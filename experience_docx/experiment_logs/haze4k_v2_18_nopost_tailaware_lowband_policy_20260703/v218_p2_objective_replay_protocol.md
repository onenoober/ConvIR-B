# v2.18 P2 Tail-Aware Objective Replay Protocol

No training is run here. This replay asks whether the proposed tail and preserve objective terms would activate on the known v2.16 WLDB-A failure mode.

Tail hinge: activates when candidate dPSNR vs A0 is below `-0.15`.
Severe hinge: audits dPSNR below `-0.20`.
Preserve hinge: activates on strong/easy samples below `-0.05`.
Action budget sweep: thresholds are derived from P1 safe oracle/predicted O1 action norms and applied to v2.16 WLDB-A action norms.
Locked Haze4K remains untouched.
