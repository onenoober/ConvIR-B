# Haze4K v5 CHD-RM v0 Route Lock

Date: 2026-07-08

Status: completed route-lock gate; runtime experiments not launched.

## Scope

- Project: ConvIR-B Haze4K.
- Model family: CHD-RM v5.
- Dataset or task: Haze4K single-image dehazing.
- Primary objective: lock research content one before any architecture or
  runtime experiment.
- Main metric: not applicable at v0; later stages use paired dPSNR and
  region/tail gates.
- Secondary metrics: SSIM, LPIPS, calibration, gamma behavior, efficiency.
- Execution environment: runtime validation only on `convir-4090`.
- Artifact root:
  `experience_docx/experiment_logs/haze4k_v5_chd_rm_v0_route_lock_20260708/`.
- Branch or isolated workspace:
  `codex/haze4k-v5-v0-chd-rm-route-lock`.

## Fact Sources

- GitHub main rules:
  `experience_docx/MODEL_EXPERIMENT_START_CHECKLIST.md`,
  `experience_docx/OFFICIAL_ARCH_ANCHOR_POLICY.md`,
  `experience_docx/BRANCH_EXPERIMENT_SYNC_PROTOCOL.md`.
- User route input:
  `研究内容一最终研究路线`, provided on 2026-07-08.
- Runtime source for future raw outputs: `convir-4090`.

## Route Identity

This is a new model-structure route, not a continuation, rescue, ablation, or
evidence sync of earlier Haze4K routes.

Consequences:

- source branch must be the official ConvIR-B architecture anchor;
- old failed branch code must not be inherited;
- locked test remains closed until v7 candidate lock;
- architecture changes are allowed only if they keep the CHD-RM research
  content fixed.

## Source Contract

| Field | Value |
| --- | --- |
| Source branch | `github/codex/haze4k-official-arch-anchor` |
| Source commit | `3b4da35440c8c26a7d1bcaf1daf342e11d9a3898` |
| Local route branch | `codex/haze4k-v5-v0-chd-rm-route-lock` |
| Cloud workspace | `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v0-chd-rm-route-lock` |
| Cloud Python | `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python` |
| Dataset root | `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K` |
| Baseline checkpoint | `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl` |
| Locked-test root | `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K-locked-test-flat-1000-v210-20260616` |

## Fixed Research Content

```text
连续雾浓度感知的区域自适应残差调制与低雾区域保护去雾方法研究
```

Out of scope:

- independent color/luminance/texture/structure fidelity modules;
- color correction, texture enhancement, or structure preservation modules;
- Lab/luminance/gradient/texture as core training targets;
- backbone replacement;
- test-driven threshold, checkpoint, scale, gamma, mask, or architecture
  selection.

## Stage Registry

| Stage | Registered Branch | Gate |
| --- | --- | --- |
| v0 route lock | `codex/haze4k-v5-v0-chd-rm-route-lock` | fixed route scope and test policy |
| v1 data baseline | `codex/haze4k-v5-v1-chd-rm-data-baseline-lock` | data, leakage, A0, metric stability |
| v2 density need | `codex/haze4k-v5-v2-chd-rm-density-need-calibration` | density/need calibration |
| v3 no-op RARM | `codex/haze4k-v5-v3-chd-rm-noop-rarm-audit` | A0 equivalence and cost |
| v4 single-scale | `codex/haze4k-v5-v4-chd-rm-single-scale-rarm` | matched-budget superiority |
| v5 low-haze protection | `codex/haze4k-v5-v5-chd-rm-low-haze-protection` | low-haze protection |
| v6 multiscale | `codex/haze4k-v5-v6-chd-rm-multiscale-haze-modulation` | value over v5 and cost |
| v7 OOF lock | `codex/haze4k-v5-v7-chd-rm-oof-candidate-lock` | one fixed candidate |
| v8 final confirmation | `codex/haze4k-v5-v8-chd-rm-final-haze4k-confirmation` | one-shot locked test |

## Forbidden Flows

- Do not start from any APDR, DPGA, NoPost, SFAD, DCFSB, A0Prox, ConvIR-WD, or
  other previous route branch.
- Do not use Haze4K locked test before v8.
- Do not tune from locked-test results.
- Do not skip v2 and directly attach RARM.
- Do not train RARM if v3 no-op equivalence fails.
- Do not promote a route that only beats A0 but fails matched-budget controls.
- Do not claim final confirmation from canary, val600, or OOF alone.

## v0 Decision

Route lock passes. The next allowed work is v1 data and ConvIR-B baseline
locking on `convir-4090`, with locked test closed and no model changes beyond
documented v1 support scripts.
