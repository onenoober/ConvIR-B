# Haze4K v2.2 C8-Mini Multi-Expert Complementarity Evidence

Status: `C8_STOP_PREFLIGHT_FAILED_ENGINEERING_ASSET_UNAVAILABLE`

Route card: `experience_docx/experiment_cards/2026-06-15-haze4k-v2-2-c8-mini-expert-oracle.md`

## Runtime Contract

- Host: `convir-4090` only.
- Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c8-mini-expert-oracle`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Haze4K data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- Split JSON: `experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604/internal_val/haze4k_dpga_v13_regular_hard_seed3407.json`.
- Locked test: untouched and forbidden for C8.

## Phases

- C8-0: preregistration, expert manifest, script hash, asset/download probe, no-locked status.
- C8-1: WDMamba single-expert audit if checkpoint/result assets are reproducible.
- C8-2: FSNet+UDP duplicate audit and conditional join if non-duplicate with assets.
- C8-3: MB-TaylorFormerV2-L fallback if preceding candidates are unavailable or insufficient.
- C8-Decision: decide whether C9 router is scientifically justified; no router is trained here.

## Status Files

- `status_c8_0.txt`
- `status_c8_1.txt`
- `status_c8_2.txt`
- `status_c8_3.txt`

## Result

Decision: `C8_STOP_PREFLIGHT_FAILED_ENGINEERING_ASSET_UNAVAILABLE`

- C8-0 preregistration and asset audit completed.
- C8-1 WDMamba stopped before rendering because no Haze4K checkpoint/result package is available.
- C8-2 FSNet+UDP duplicate audit found source files are not identical to current ConvIR+UDP, but no FSNet+UDP Haze4K checkpoint is available for render-diff duplicate proof.
- C8-3 MB-TaylorFormerV2-L fallback stopped before rendering because no Haze4K-L checkpoint is available.
- Locked test remained untouched; no router/MoE/distillation is authorized.

Primary artifacts: `v22_c8_0_expert_manifest.json`, `v22_c8_2_fsudp_duplicate_audit.md`, `v22_c8_decision.md`, and `v22_c8_summary.json`.
