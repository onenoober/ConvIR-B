# Haze4K v2.1 SEG-Mix Multi-Alpha / Local-Alpha Evidence

Status: `C6_SCREEN_PASS_STRONG_TARGET_NOT_YET_C7_PATCH_SIGNAL`

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
- C7b: train-derived local-alpha deployable prototype using image-fold OOF and true held-out PSNR/SSIM re-render.
- C7c: severe-risk tightening profiles using C7b patch feature/SSE rows and one true held-out re-render pass.
- C9: profile-level shifted strong validation over train-derived stress bins.
- C9b: fixed conservative profile stress for `riskcap36_no075`.
- C10: formal 5x3 fixed-profile replay; locked authorization only if strong formal gate passes.

## Status Files

- `status_c5.txt`
- `status_c6_c7.txt`
- `status_c7b.txt`
- `status_c7c.txt`
- `status_c9.txt`
- `status_c9b.txt`
- `status_c10.txt`

This README will be updated after cloud evidence is synced back from `convir-4090`.

## Results

Decision: `C6_MULTIALPHA_OOF_SCREEN_PASS_STRONG_TARGET_NOT_YET_START_C7_C8__C7_PATCH_ALPHA_ORACLE_STRONG_SIGNAL_START_LOCAL_ALPHA`

C5 completed forensic replay without policy tuning:

- hard-bottom25 rows with an existing safe high-alpha candidate: `97/150`.
- seeded positive deficits to 0.70: `[11, 19, 6]`.
- seeded selected-negative counts: `[97, 93, 98]`.

C6 exact multi-alpha OOF router:

- mean `+0.422839 dB`.
- hard bottom-25 `+0.479300 dB`.
- easy top-25 `+0.447305 dB`.
- dSSIM `+0.00027525`.
- positive ratio `0.698333`.
- severe regressions `46.0/600`.
- screen gate `True`, strong-candidate gate `False`.

Image-level multi-alpha oracle remains strong:

- mean `+0.828900 dB`, hard `+0.926646 dB`, positive `0.796667`, severe `0.0/600`.

C7 patch-alpha oracle has strong local-alpha signal:

- max oracle: mean `+1.160954 dB`, hard `+1.107960 dB`, positive `0.995000`, severe `0.0/600`.
- risk-capped oracle: mean `+0.876923 dB`, hard `+0.756983 dB`, positive `0.995000`, severe `0.0/600`.

## Decision

C6 does not authorize C9/C10 or locked because positive ratio remains below `0.70`. C7 authorizes a train-derived local-alpha prototype. Locked test and distillation remain blocked.

## C7b Result

Decision: `C7B_LOCAL_ALPHA_FAIL_START_C8_MULTIEXPERT_OR_RICHER_LOCAL_FEATURES`

C7b train-derived local-alpha deployable prototype re-rendered held-out images for true PSNR/SSIM:

- mean `+0.376111 dB`.
- hard bottom-25 `+0.360949 dB`.
- easy top-25 `+0.443171 dB`.
- dSSIM `+0.00025762`.
- positive ratio `0.793333`.
- severe regressions `50.0/600`.

C7b fails only the severe gate (`50/600` > `48/600`). This authorizes one train-derived C7c severe-risk tightening pass; it does not authorize C9/C10, locked test, or distillation.

## C7c Result

Decision: `C7C_RISK_TIGHTEN_STRONG_PASS_START_C9_SHIFTED_STRONG`

Best strong profile: `riskcap42_no075`.

- mean `+0.354799 dB`.
- hard bottom-25 `+0.322247 dB`.
- easy top-25 `+0.451988 dB`.
- dSSIM `+0.00024897`.
- positive ratio `0.790000`.
- severe regressions `43.0/600`.

C7c passes the strong train-derived OOF gate and authorizes C9 shifted-strong validation only. Locked test and distillation remain blocked.

## C9 Result

Decision: `C9_SHIFTED_STRONG_FAIL_REASSESS_LOCAL_ALPHA_OR_C8`

C9 profile-level shifted strong validation passed 8/9 dimensions. The only failing dimension was `diff_signed_q4` with severe `50.0/600`, two above the `48/600` gate. This does not authorize C10. A C9b fixed conservative profile stress using the predeclared C7c `riskcap36_no075` profile is authorized to determine whether the failure is profile-selection instability. Locked remains blocked.

## C9b Result

Decision: `C9B_FIXED_PROFILE_SHIFTED_PASS_START_C10_FORMAL_5X3`

Fixed profile `riskcap36_no075` passed all shifted stress dimensions: mean `+0.341530`, hard `+0.310932`, positive `0.786667`, severe `37.0/600`. C10 formal 5x3 is authorized. Locked remains blocked until C10 passes and the route card is updated.
