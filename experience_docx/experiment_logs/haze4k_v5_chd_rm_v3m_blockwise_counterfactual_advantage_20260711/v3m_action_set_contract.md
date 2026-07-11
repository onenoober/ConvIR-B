# v3m-A0a Action Set Contract

Deployment-comparable oracle ladder:

```text
{0, 0.125, 0.25, 0.5, 1.0}
```

Image, block16, block32, and pixel-grid policies all select only from this
ladder. The A0a denominator is pixel-grid, not continuous pixel alpha. Dense
`1/32` and continuous pixel results are deferred to A0b as mechanism-only
diagnostics and cannot rescue A0a.
