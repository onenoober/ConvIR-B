# v2i Route Decision

Status: `PLANNED`

Decision label: `PLANNED_V2I_FAM2_NOOP_ARCH_EQUIVALENCE_AUDIT`

v2h A/B/C established D7c as a deployable actionable prior candidate, while
v2h-D correctly stopped at the official-anchor architecture boundary. Therefore
the next route is a separate model-structure no-op audit from the official
anchor, not an in-place v2h mutation and not RARM/training.

v2i may only answer this question:

```text
Can FAM2-only zero-init modulation be inserted from the official ConvIR-B anchor
while preserving exact A0 behavior?
```

It does not decide whether D7c-gated modulation or RARM training improves image
quality. Those remain blocked until no-op equivalence passes.
