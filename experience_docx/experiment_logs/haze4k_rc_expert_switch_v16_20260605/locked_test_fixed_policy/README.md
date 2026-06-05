# Haze4K v1.6 RCExpertSwitch Locked Test

Status: `LOCKED_TEST_COMPLETE`.

Decision: `LOCKED_TEST_FAIL_NO_FURTHER_SELECTION`.

Locked test touched: yes.

Fixed policy:

```text
feature = udp_a0_luma_shift_mean
direction = low
threshold = -0.003969017509371043
fallback = A0
expert = official UDPNet ConvIR
```

This directory is a one-shot confirmation for the fixed internal policy. Do not
use these results to change threshold, feature, checkpoint, or expert set.
