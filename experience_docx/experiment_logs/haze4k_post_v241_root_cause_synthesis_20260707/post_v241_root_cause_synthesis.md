# Post-v2.41 Root-Cause Synthesis

Decision: `TAIL_SAFE_INCREMENT_CONTROLLABILITY_FAIL`.

Updated finding: v2.41 removed teacher targets and selector dependence, started from the official ConvIR-B anchor, passed strict Stage-0 identity/preflight, and trained only a bounded A0-proximal residual head. The canary32 OOF gate still failed: mean/hard/easy `-0.0277/+0.0742/-0.0724 dB`, p05/CVaR5 `-0.3981/-0.5972 dB`, severe regressions `27`, strong-reference regressions `25`, and fold pass `0/5`.

v2.42 then recomputed the v2.41 OOF table exactly and decomposed the failure. The failure is direction-dominant: all `27/27` severe rows are `direction_bad`, none are `overshoot_bad`, no global shrink gamma passes the gate, oracle clamp upper bound is weak, and train32 full-image evaluation also fails.

Interpretation: the blocker is not forbidden postprocessing, checkpoint loading, teacher leakage, lack of residual-energy concentration on hard images, a simple scale issue, or train/OOF split overfit alone. The current frozen ConvIR-B plus tiny A0-proximal residual head cannot reliably learn GT-improving residual direction around a strong A0 baseline.

Do not reopen v2.41 by more epochs, folds, samples, loss-weight tuning, beta-only shrink, canary80, locked test, WDMamba/ConvIR-L selector or alpha continuation, bridge/generator work, or P5 projection.

Allowed future work must be a materially changed route, not a v2.41 rescue. Candidate direction: A0-anchored partial-unfreeze or larger GT-risk-controlled ConvIR route with fresh Stage-0 preflight and OOF-first gates.
