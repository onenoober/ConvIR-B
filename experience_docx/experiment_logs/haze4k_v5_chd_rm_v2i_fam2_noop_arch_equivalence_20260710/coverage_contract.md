# v2i Coverage Contract

v2i itself does not select or resize D7c masks. It records this contract because
v2h showed different coverage views across risk-coverage and shadow-modulation
audits.

- v2h-A `selected_coverage` is score-mask coverage on the val-inner tensor view
  used by the D7c three-state target.
- v2h-B `selected_coverage` is the shadow-modulation valid-pixel view after
  applying the selector mask to A0 residual oracle pixels.
- v2i no-op equivalence does not use coverage for pass/fail.
- A later D7c-gated no-op route must explicitly state mask source, resize mode,
  FAM2-scale coverage, image-scale coverage, boundary handling, low-haze touch,
  negative-low-risk touch, and per-image tail gates before launch.
