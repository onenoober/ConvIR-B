# Haze4K v2.1 SEG-Mix Multi-Alpha / Local-Alpha Evidence

Status: `PLANNED_C5_C6_C7_NO_LOCKED`

Route card: `experience_docx/experiment_cards/2026-06-15-haze4k-v2-1-segmix-multialpha-local.md`

## Runtime Contract

- Host: `convir-4090` only.
- Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v21-segmix-multialpha-local`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Remote copy fallback: if GitHub clone/fetch is unavailable on `convir-4090`, sync this committed branch by `git archive` and write `.codex_source_branch`, `.codex_source_commit`, and `.codex_source_copy_time` in the runtime workspace.
- Locked test: blocked and untouched.

## Planned Phases

- C5: C4 failure forensic, text-only replay, no policy tuning.
- C6: exact multi-alpha OOF router using a single A0/FullUDP render pass.
- C7: patch-level alpha oracle from the same render pass.

## Status Files

- `status_c5.txt`
- `status_c6_c7.txt`

This README will be updated after cloud evidence is synced back from `convir-4090`.
