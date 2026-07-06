# v2.34 P0C Metric-Contract Interpretation

Revised conclusion:

- P0/P0B failed the direct WDMamba-on-crop teacher contract.
- P0C showed the old full-image WDMamba/WD0375 table evidence remains valid for
  the full-image/full-image-output-slice contract.

What is closed:

- direct WDMamba-on-256-crop teacher canaries;
- v2.34 P1/P2/P3/P4 from direct-crop canaries;
- canary80 and locked test.

What remains open:

- full-image expert output cache followed by crop slicing;
- same-context crop teacher if eligibility is computed in the exact same crop
  context;
- context-size student contract audit.

New unresolved issue:

Full-image teacher slice must be rebased against the actual student baseline.
If the student remains 256 crop-input, compute:

```text
T_fullslice_PSNR - A0_cropdirect_PSNR
```

before authorizing any projection or training.
