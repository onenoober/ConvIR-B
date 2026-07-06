# v2.37 Decision Tree

P0 alpha sweep:

- If an unmasked alpha passes the full600 gate, stop this mask audit before
  bridge/generator work and record the passing alpha as the clean lower-alpha
  substrate candidate.
- If no alpha passes, proceed to P1/P2 masked-preservation audit.

P2 mask sweep:

- If no mask preserves enough mean/hard benefit while satisfying all tail gates,
  close the current WDMamba alpha/blend substrate.
- If at least one mask passes, proceed to P3 fold-stable mask selection.

P3 OOF mask:

- If the fold-stable substrate fails, do not train bridge/generator; change the
  teacher/alpha/context contract.
- If it passes, proceed to P4 target-only no-op separability.

P4 target-only separability:

- If unsafe/no-op is not separable from target-only features, do not train a
  bridge; require a new no-op signal or a more conservative hard-only route.
- If it passes, P5 masked free-tensor projection is authorized.

P5 masked free-tensor:

- If S4+S6 is not representable, the masked full600 substrate is inconsistent
  with v2.35 canary representability.
- If S4+S6 is representable, open a separate v2.38 masked WLFBridge-S4S6
  generator trainability route.
