# v3n Deviation Log

## 2026-07-12 A0 r0 engineering failure

The first A0 launch failed before producing a scientific result because the
script referenced `args.max_fold_negative_false_rate_per_fold`; the parser
defines `args.max_negative_false_rate_per_fold`. The corrected r1 script is
route commit `a76318f25afbb61dce52d700d3a79f3f8143a6dd` and is the only valid
A0 result.
