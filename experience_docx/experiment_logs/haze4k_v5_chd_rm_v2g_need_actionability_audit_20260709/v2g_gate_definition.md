# v2g Gate Definition

Policy gates:

- locked Haze4K test usage: `none`.
- D2: `not_run`.
- RARM: `not_connected_or_trained`.
- v3: `not_run`.
- ConvIR-B A0 and D3 density are treated as frozen prior evidence only.

G0 must reproduce the core v2d/v2e/v2f/F4b numbers from cloud evidence:

- D7c top-k safety/ranking and LDHN recall.
- D7c-RP recall versus false-tail tradeoff.
- F4/F4b selected failure and best safe LDHN recall.
- No locked-test usage.

G1 must report LDHN semantics:

- LDHN total support, core/boundary/adjacent-to-haze/isolated/unstable fractions.
- Per-image recall and miss-rate distributions for D7c on each LDHN subtype.
- Connected-component and blur-stability summaries.
- Interpretation of whether global LDHN is a safe RARM action target.

G2 is diagnostic only:

- Current deployable-information upper bound from existing feature probes.
- Physics-oracle asset availability audit; if assets are missing, report blocker.
- Residual-oracle availability audit; if assets are missing or would require locked test, report blocker.

No model-promotion gate exists in v2g. A later route may define actionable LDHN targets only after v2g closeout.
