# v2.35 Decision Tree

P0D asks whether a full-image-output teacher slice is better than the actual
256 crop-input student baseline, crop-direct A0.

If P0D passes for an alpha:

- crop-input student plus full-image-slice target can remain under audit;
- continue P1/P2/P3;
- no training is authorized yet.

If P0D fails:

- do not train a 256 crop-input student on a full-image-slice target;
- continue P1 full-image cache audit and P2 context-size sweep;
- only larger-context or full-image contracts may remain viable.

P2 asks whether any context size makes the full-image-slice teacher positive
against the same-context A0 baseline.

If P2 passes:

- adopt the passing context/alpha as the next student/baseline contract;
- run P3 same-contract positive-substrate manifest.

If P2 fails:

- WDMamba remains valid full-image evidence but is not a current NoPost student
  target;
- stop before P3/P4, bridge training, canary80, and locked test.

P3 asks whether the passing context has enough tail-safe positive samples.

If P3 passes:

- P4 same-contract free-tensor projection can be proposed in a separate written
  gate/command.

If P3 fails:

- stop before P4, bridge training, canary80, and locked test.
