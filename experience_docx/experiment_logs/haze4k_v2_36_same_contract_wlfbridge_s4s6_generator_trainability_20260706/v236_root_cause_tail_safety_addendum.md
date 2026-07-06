# v2.36 Root-Cause Tail-Safety Addendum

Final route decision:
`P0_FAIL_STOP_BEFORE_BRIDGE_TRAINING`.

## Interpretation

The full-image same-context WDMamba alpha0.5 teacher is strongly positive on
mean/hard/easy, but the unmasked full600 substrate is not tail-safe. The route
failed because CVaR5, severe regressions, strong-reference regressions, and fold
gates failed.

## Not Evidence Of

- Bridge/generator failure.
- S4+S6 representability failure.
- Full-image WDMamba teacher invalidity.

## Evidence Of

- The unmasked alpha0.5 full600 substrate is not a valid training target.
- A new route must first define and pass a tail-safe eligibility/no-op/
  preservation contract before any bridge or generator is authorized.

## Blocked

P0B context384 projection, P1 architecture identity, P2 generator fit, P3 OOF,
canary80, and locked test are blocked by the P0 failure.
