# Haze4K Architecture Fine-Tune Workflow

Date: 2026-07-16

This file contains only Haze4K architecture-specific rules. Use the repository
start checklist, governance and run protocol for the universal workflow.

## Fixed Source And Runtime

- Source: immutable `github/codex/haze4k-official-arch-anchor`.
- Cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- Official checkpoint:
  `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.

Create a fresh `codex/<route>` branch/workspace from the exact anchor commit.
Do not start from an experimental leaf unless the new route explicitly tests an
inherited intervention and records that predecessor as a factor/reference.

## Additive Architecture Contract

- Put new parameters under route-specific prefixes.
- Prefer a route-specific wrapper/builder and `--arch` value. Preserve the
  official default entrypoint and three-scale/checkpoint/padding contracts.
- Do not change an official pretrained tensor shape and still claim it was
  reused. List every affected tensor as newly initialized.
- Initialize residual heads to zero, FiLM-like modulation to identity, and
  gates/routers to a conservative no-op with a fallback.
- The integrated preflight proves exact no-op equivalence or a predeclared
  bounded difference before expensive work.

## Strict Partial Load

When reusing the official checkpoint:

- every matching official key loads with exact shape;
- missing keys are allowed only under the declared new prefixes;
- unexpected checkpoint keys and official-key shape mismatches are fatal;
- record checkpoint path/size/SHA-256 plus loaded, allowed-missing, unexpected
  and shape-mismatch lists/counts;
- never use broad `strict=False` or wildcard missing-key allowances.

After assembling the accepted state, load it with `strict=True`. If a trusted
legacy checkpoint needs `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, record that fact
in status and compact evidence.

## Trainable Scope

Choose the smallest informative scope before results:

- `adapter_only`: route prefixes only; frozen backbone stays in eval mode;
- `adapter_neighbor`: route prefixes plus named adjacent layers;
- `selected_backbone`: route prefixes plus named backbone stages;
- `all`: only with a written reason smaller scopes cannot answer the mechanism.

Do not force a universal scope/LR/epoch ladder. Freeze optimizer groups, budget,
weight decay and gradient policy from the matched contract. A wider scope is a
new operation requiring a prior closeout that names that exact operation id.

## Required Architecture Checks

The runner records parameter counts by scope, strict partial-load results,
synthetic output shapes, finite forward, no-op/bounded difference, one native
real-batch finite forward/backward when training, route activity, and
`locked_test_touched=false`. Formal evaluation uses the same data decoding,
padding and metrics as the matched reference, with a primary paired/grouped
effect, tail/strong-reference preservation and one mechanism metric. Add
latency/memory only when gated.

Locked Haze4K test cannot select checkpoints, scope, scale, threshold or active
modules. Mean PSNR alone never promotes an architecture.
