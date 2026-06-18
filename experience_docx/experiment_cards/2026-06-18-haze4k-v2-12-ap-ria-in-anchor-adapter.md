# Haze4K v2.12 AP-RIA In-Anchor Adapter Route Card

Status: `DRAFT_CODE_ONLY_NO_LOCKED_ACCESS`

## Branch

Recommended branch name:

```bash
codex/haze4k-v2-12-ap-ria-in-anchor-adapter
```

Starting anchor:

```text
github/codex/haze4k-official-arch-anchor @ 2d529d4
```

## Motivation

Recent v2.10/v2.11 locked diagnostic evidence shows that strong expert endpoints and residual-shrinkage alpha curves have a consistent gain-risk tradeoff:

- WDMamba full improves mean/hard but has much worse positive ratio and severe-tail than moderate shrinkage.
- FSNet+UDP and MB-TaylorFormerV2-L show the same broad pattern: aggressive/full expert use improves some hard samples but damages coverage and tail safety.

AP-RIA moves the residual-calibration idea inside the anchor network:

```text
F' = F + G_low * ΔF_low + G_detail * ΔF_detail
O  = Head_A0(F') + I
```

It does not use `E - A0` as a runtime input or output-level fusion signal.

## Scope

- Anchor: official ConvIR-B A0.
- Insertion: after `Decoder[2]`, before `feat_extract[5]`.
- Trainable: AP-RIA adapter only in first pass.
- Frozen: ConvIR-B anchor and all experts.
- Teacher: optional training-only guidance bank from WDMamba / FSNet+UDP / MB-TaylorFormerV2 outputs.
- Inference: ConvIR-B + AP-RIA only. No expert, no `E-A0`, no GT.

## Checkpoint Loading And Initialization

When fine-tuning from the official Haze4K checkpoint
`/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`:

- matching official ConvIR-B keys must load exactly;
- unexpected checkpoint keys are fatal;
- shape mismatches in official keys are fatal;
- allowed missing/new prefix: `ap_ria.*`;
- `ap_ria.zero_proj.weight` and `ap_ria.zero_proj.bias` are zero-initialized, making the adapter path an identity/no-op at construction;
- evidence/gate/correction submodules under `ap_ria.*` use their default PyTorch initializers, but their output cannot change the restored image before `zero_proj` is trained;
- first-pass training freezes official ConvIR-B parameters and trains only `ap_ria.*`.

## Required preflight

1. Identity initialization audit:
   - `max_abs(output_0 - A0_side) <= 1e-6`.
2. Trainable parameter audit:
   - only `ap_ria.*` parameters trainable in the first pass.
3. Runtime evidence audit:
   - evidence comes only from `I`, `A0_side`, and internal feature `F`.
4. Locked policy:
   - no locked test read or tuning during development.

## First-pass ablations

1. Full AP-RIA.
2. w/o teacher guidance.
3. w/o runtime evidence bank.
4. w/o low/detail split.
5. w/o risk-aware gates.
6. low-only.
7. detail-only.
8. w/o anchor preservation.
9. random-init projection vs zero-init projection.

## Metrics

Report mean PSNR/SSIM plus reliability metrics:

```text
mean dPSNR
hard bottom-25 dPSNR
easy top-25 dPSNR
positive ratio
severe regression count
worst-5% dPSNR
dSSIM
gate statistics
injection energy
```
