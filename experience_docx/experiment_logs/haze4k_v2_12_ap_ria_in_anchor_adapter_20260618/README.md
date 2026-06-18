# Haze4K v2.12 AP-RIA In-Anchor Adapter

Status: `DRAFT_CODE_ONLY_NO_EXPERIMENT_RUN`

This directory is reserved for the AP-RIA route:

```text
AP-RIA = Evidence-Guided Anchor-Preserving Residual Injection Adapter
```

Key code files:

```text
Dehazing/ITS/models/AP_RIAConvIR.py
experience_docx/tools/ap_ria_loss_utils.py
experience_docx/tools/smoke_ap_ria_model.py
```

Important distinction:

```text
Old output-level form:
    O = A0 + alpha * (E - A0)

New in-anchor form:
    F' = F + G_low * ΔF_low + G_detail * ΔF_detail
    O  = Head_A0(F') + I
```

Teacher outputs are training-only guidance sources. They are not AP-RIA runtime inputs.
