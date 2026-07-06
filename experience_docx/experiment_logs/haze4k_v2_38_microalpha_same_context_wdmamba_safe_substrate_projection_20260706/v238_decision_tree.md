# v2.38 Decision Tree

P0:

- If no micro-alpha passes the strict full600 and fold safety gates, close the
  unmasked WDMamba substrate route and choose between v2.38B richer target-only
  no-op/unsafe separability or ConvIR-L same-family teacher audit.
- If at least one alpha passes, proceed to P1 OOF alpha selection.

P1:

- If alpha only passes when selected on all 600 but fails OOF selection, treat
  it as an overfit alpha threshold and do not project.
- If OOF selection passes all heldout folds with stable selected alpha, proceed
  to P2 safety-margin audit.

P2:

- If selected alpha has poor safety margin, lower alpha only if the written P1
  selection rule still authorizes it; otherwise stop.
- If safety margin passes, authorize P3 unmasked micro-alpha free-tensor
  projection.

P3:

- If S4+S6 free-tensor projection passes on the selected micro-alpha, open a
  separate v2.39 micro-alpha WLFBridge-S4S6 generator trainability route.
- If projection fails, record that the micro-alpha target is safe but not
  representable enough in the current internal carrier.
