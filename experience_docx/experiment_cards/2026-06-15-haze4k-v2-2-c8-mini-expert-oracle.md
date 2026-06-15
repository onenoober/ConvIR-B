# Haze4K v2.2 C8-Mini Train-Derived Multi-Expert Complementarity Oracle

Date: 2026-06-15

Status: `C8_STOP_PREFLIGHT_FAILED_ENGINEERING_ASSET_UNAVAILABLE`

## Scope

- Objective: prove whether new experts complement the v2.1 FullUDP/local-alpha hard-domain blind spot before any MoE/router training.
- Runtime host: `convir-4090` only.
- Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c8-mini-expert-oracle`.
- Evidence root: `experience_docx/experiment_logs/haze4k_v2_2_c8_mini_expert_oracle_20260615/`.
- Branch: `codex/haze4k-v2-2-c8-mini-expert-oracle`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.

## Locked-Test Contract

C8-Mini is train-derived only. It must not read locked per-image outputs, tune from locked evidence, choose thresholds/features/checkpoints/actions from locked evidence, train a router, train MoE, distill, or run locked Haze4K. The v2.1 locked one-shot is consumed and failed; it is evidence-only.

## Fixed Candidate Order

```text
S0: A0 + current FullUDP / v2.1 local-alpha family
S1: S0 + WDMamba
S2: S1 + FSNet+UDP after duplicate audit only
S3 fallback: S2 + MB-TaylorFormerV2-L only if WDMamba/FSNet+UDP is unavailable or insufficient
```

Excluded from C8: DEA-Net, CasDyF-Net, DehazeXL, PromptIR/AdaIR, MoE/router training.

## Fixed Alpha Grid

```text
{0, 0.0625, 0.125, 0.25, 0.375, 0.50}
```

No alpha `0.75/1.0` in the first deployable candidate.

## Required Outputs

C8-0:

- `v22_c8_0_expert_manifest.json`
- `v22_c8_0_metric_script_sha256.txt`
- `v22_c8_0_no_locked_status.txt`
- `v22_c8_0_download_probe.csv`
- `v22_c8_0_reliability_note.md`

C8-1 WDMamba:

- `v22_c8_1_wdmamba_single_summary.csv`
- `v22_c8_1_wdmamba_alpha_grid.csv`
- `v22_c8_1_wdmamba_oracle_vs_s0.csv`
- `v22_c8_1_wdmamba_group_metrics.csv`
- `v22_c8_1_wdmamba_unique_wins.csv`
- `v22_c8_1_wdmamba_decision.md`

C8-2 FSNet+UDP:

- `v22_c8_2_fsudp_duplicate_audit.md`
- `v22_c8_2_fsudp_single_summary.csv`
- `v22_c8_2_fsudp_alpha_grid.csv`
- `v22_c8_2_s2_forward_selection_oracle.csv`
- `v22_c8_2_s2_expert_composition_by_group.csv`
- `v22_c8_2_s2_decision.md`

C8-3 MB-TaylorFormerV2-L:

- `v22_c8_3_mbtaylor_single_summary.csv`
- `v22_c8_3_mbtaylor_alpha_grid.csv`
- `v22_c8_3_s3_forward_selection_oracle.csv`
- `v22_c8_3_s3_expert_composition_by_group.csv`
- `v22_c8_3_s3_decision.md`

## Decision Gates

C8 passes only if train-derived evidence shows all of:

1. at least one new expert has hard/red-flag unique wins;
2. multi-expert oracle improves over S0 FullUDP-only oracle;
3. critical-bin min hard/min positive improves;
4. easy/tail oracle does not worsen;
5. removal ablation proves non-redundancy;
6. group-min report has no hidden red flag.

Quantitative guide:

```text
hard-bottom25 oracle over S0:  >= +0.05 dB
critical-bin min hard over S0: >= +0.05 dB
positive over S0:              >= +0.02
new expert hard/red-flag wins: >= 5%
oracle severe:                 near 0
```

If assets are unavailable, close the phase as `PREFLIGHT_FAILED_ENGINEERING_ASSET_UNAVAILABLE`, not as a scientific negative result.

## C8 Result

Decision: `C8_STOP_PREFLIGHT_FAILED_ENGINEERING_ASSET_UNAVAILABLE`

C8-0 through C8-3 completed as text-only cloud asset/preflight audits. WDMamba, FSNet+UDP, and MB-TaylorFormerV2-L could not be rendered because the required Haze4K checkpoints/result packages were unavailable on `convir-4090`; Baidu shared-link probes did not yield files without authenticated transfer. Locked test remained untouched. No C9 router/MoE is authorized.
