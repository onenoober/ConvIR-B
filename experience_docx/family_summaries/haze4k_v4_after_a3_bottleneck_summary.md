# Haze4K v4 After-A3 Bottleneck Summary

Date: 2026-07-08

Status: v4 reopened as after-A3 diagnosis; current A3 implementation remains stopped.

The two fixed v4 pain points remain unchanged:

1. Spatially non-uniform haze and polluted feature transfer.
2. Low-frequency dehazing versus high-frequency detail preservation conflict.

A3 failed because the naive combination of independent SDFM and GST modules created negative interaction. The next step is v4.4 bottleneck diagnosis, followed by independent v4.5 SDC-Lite and v4.6 DCFSB-bottleneck routes only if authorized by written evidence.
