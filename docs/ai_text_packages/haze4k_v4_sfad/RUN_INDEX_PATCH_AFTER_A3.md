# Haze4K v4 After-A3 Run Index Patch

Recommended order:

1. v4.4 bottleneck diagnosis: audit-only, no training, no locked test.
2. v4.5 SDC-Lite: shared `R_1_2`, SDFM at 1/2 only, no GST at 1/2.
3. v4.6 DCFSB-bottleneck independent: bottleneck frequency module only, no FAM/skip/loss/density changes.

Do not launch A3 + density auxiliary, A3 + DCFSB, A3 longer training, or A3 seed sweep.
