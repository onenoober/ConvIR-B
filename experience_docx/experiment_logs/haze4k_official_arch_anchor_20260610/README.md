# Haze4K Official ConvIR-B Architecture Anchor Evidence

Date: 2026-06-10

Status: `SYNCED_TO_GITHUB` after push.

## Read First

- Route card: `../../experiment_cards/2026-06-10-haze4k-official-arch-anchor.md`
- Central index: `../../EXPERIMENT_INDEX.md`
- Anchor policy: `../../OFFICIAL_ARCH_ANCHOR_POLICY.md`

## Primary Files

| File | Use |
| --- | --- |
| `run_official_anchor_preflight.sh` | Durable cloud preflight script. |
| `official_anchor_preflight.log` | Cloud stdout/stderr from anchor validation. |
| `official_anchor_preflight.json` | Structured preflight result. |
| `status.txt` | Start/end status markers. |
| `source_audit.txt` | Upstream source comparison and anchor immutability decision. |

## Key Result

`OFFICIAL_ANCHOR_PREFLIGHT_OK` on `dehaze1`.

| Check | Result |
| --- | --- |
| checkpoint strict load | pass, `missing=[] unexpected=[]` |
| checkpoint sha256 | `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088` |
| official state cleanliness | pass, no FAM modulator/APDR/DPGA/PFD state keys |
| parameter count | `8,630,665` |
| synthetic forward | pass, finite outputs at 1/4, 1/2, and full scales |
| Haze4K train crop forward | pass, finite multiscale L1 `0.009162915870547295` |
| CLI compatibility | pass, both `--learning_rate` and `--leaning_rate` are accepted |
| locked test | untouched |
| source audit | pass, `ConvIR.py` upstream-equivalent except original-only wrapper; `layers.py` Haze4K-equivalent |

Decision: keep `codex/haze4k-official-arch-anchor` as the immutable official
architecture anchor. Future model architecture work must branch from this
anchor rather than modify it directly.

## Locked-Test Policy

The preflight uses synthetic input and one Haze4K train crop. It must not touch
Haze4K test or perform checkpoint/threshold selection.
