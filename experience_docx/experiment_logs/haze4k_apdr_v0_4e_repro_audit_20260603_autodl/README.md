# Haze4K APDR-v0.4E AutoDL Fixed-Code Repro Audit

Date: 2026-06-04

Status: `826caaf` clean-code rerun completed enough to confirm the stop
direction, but it exposed two additional reproducibility issues: variable-schema
CSV writing and `kernel/kenel` mapper-name compatibility. No E2, full router,
local correction, dense residual head, or stop20 is authorized.

## Provenance

- Server: `autodl-dehaze4`.
- Clean checkout path:
  `/root/autodl-tmp/workspace/ConvIR-B-v04e-repro-826caaf-bundle`.
- Checkout method: local pushed commit was exported as a complete git bundle
  because AutoDL GitHub clone/fetch stalled; the bundle recorded
  `826caaf0031cb46cb925b00341eba0504b397c1b`.
- Dataset:
  `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`.
- Selector:
  `/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-2rc-oracle-diagnostic/experience_docx/experiment_logs/haze4k_apdr_v0_2rc_frozen_residual_20260603/selector_checkpoint_apdr_v0_2rc_frozen_selector_seed3407.pkl`.

## Environment Snapshot

`autodl_min_repro_audit_826caaf.txt` records:

- `git rev-parse HEAD`: `826caaf0031cb46cb925b00341eba0504b397c1b`.
- `git diff --stat`: empty before rerun.
- Python: `3.10.13`.
- PyTorch: `2.11.0+cu128`.
- `F.interpolate` signature uses `align_corners`, confirming the original
  `align_coners` spelling would not be accepted.
- GPU at audit start: `NVIDIA GeForce RTX 4090`.

## Rerun Outcome

- E0 risk action bank finished and wrote a locked-threshold summary.
- E1 OOF finished the expensive 3000-image OOF action evaluation and wrote
  the per-image intermediate CSV, but the original script failed during
  variable-field CSV writing.
- A finalize script was added and used to regenerate missing E1 summary tables
  from the per-image intermediate CSV without rerunning GPU evaluation.
- Policy search found `0` retained gate-passing policies.

## Additional Implementation Findings

The fixed `826caaf` rerun still did not fully reproduce the older 60000-row OOF
table because the v0.4D probe emits `*_kernel_knn_9` mapper names while v0.4E
default candidates and historical locked rules used `*_kenel_knn_9`. This
filtered out the three KNN mapper families and left only:

- `global_mean_coeff`
- `spatial_priors_ridge_10`

The current code now adds a compatibility layer that accepts both
`kernel/kenel` names and writes `kernel_confidence/kenel_confidence` aliases.
An alias-corrected full OOF rerun is still required for exact numeric sealing,
but the current route remains stopped because the reproduced RuleB and policy
search failure match the older stop direction.
