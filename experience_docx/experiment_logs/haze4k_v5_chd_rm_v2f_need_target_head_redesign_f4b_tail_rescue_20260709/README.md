# CHD-RM v2f F4b Tail-Rescue Matrix Evidence

Status: `COMPLETED_GATE_FAIL`

Decision label: `PAUSE_V2F_F4B_NO_SAFE_LDHN_POINT_NO_F5_NO_V3`

Route card: `experience_docx/experiment_cards/haze4k-chd-rm-v2f-need-target-head-redesign.md`

Central index: `experience_docx/CHD_RM_EXPERIMENT_INDEX.md`

Purpose: test whether the v2f F4 gate failure was caused by insufficient low-density hard-negative tail pressure rather than by a deeper target/head separability limit.

Policy: ConvIR-B A0 and D3 density stayed frozen. D2, v3, RARM connection/training, and locked Haze4K test were not run. The original v2e global safety/LDHN gate remained the decision contract.

## Primary Evidence

- authorization: `f4b_authorization_record.md`
- status: `status.txt`
- closeout: `v2f_f4b_tail_rescue_closeout.json`
- summary: `v2f_f4b_tail_rescue_summary.md`
- matrix table: `f4b_tail_rescue_matrix_summary.csv`
- per-spec evidence roots: `tail2_topk10/`, `tail3_cap128_temp04/`, `tail4_topk20/`

## Key Result

F4b completed all three declared specs and found no selected gate pass and no safe+LDHN threshold point. Best safe LDHN recall was only `0.0523`. The variants that reached high LDHN recall had false-p95 near `0.9895` to `1.0000`; the safest selected global point had false-p95 `0.0246` but LDHN recall only `0.0262`.

## Decision

Keep v2f paused. Do not run F5, v3, RARM, D2, ConvIR-B unfreeze, or locked Haze4K test from this state. Do not repeat F4/F4b strength sweeps without a new written route decision that changes target semantics or available information.

Runtime metadata:

- Host: `convir-4090` / `RTX4090`
- Branch: `codex/haze4k-v5-v2f-chd-rm-need-target-head-redesign`
- Source commit: `044b77989c3afb03a06774c6e565431a50b242cd`
- Cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`
- Completed: `2026-07-09T19:05:55+08:00`
