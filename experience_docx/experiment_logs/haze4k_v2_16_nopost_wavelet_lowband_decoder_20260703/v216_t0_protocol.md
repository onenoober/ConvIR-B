# v2.16 T0/T1 Protocol

Route: `codex/haze4k-v2-16-nopost-wavelet-lowband-decoder`

This is a no-training, no-locked-test diagnostic for the NoPost-WLDB route.
T0 and T1 are executed in one cloud script because T0 risk-vs-lowband decoupling needs the T1 LL-oracle labels.

Forbidden: A0/WD0375/teacher/expert outputs as model forward inputs, output-output deltas as deployable features, learned RGB post-correction, locked Haze4K.

Allowed here: train-derived A0 prediction and GT are used only for oracle/headroom measurement, not as a deployable forward contract.
