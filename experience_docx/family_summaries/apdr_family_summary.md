# APDR Family Summary

Date: 2026-06-04

Status: broad output-residual and coefficient-mapping forms are stopped; safe
subset/action-bank ideas require fixed-code OOF or held-out evidence before any
long run.

## Sources

- Index: `../EXPERIMENT_INDEX.md`
- Cards:
  - `../experiment_cards/2026-06-02-haze4k-apdr-convir-v0.md`
  - `../experiment_cards/2026-06-02-haze4k-apdr-convir-v0-1.md`
  - `../experiment_cards/2026-06-02-haze4k-apdr-convir-v0-2-selector.md`
  - `../experiment_cards/2026-06-02-haze4k-apdr-convir-v0-2r-selector.md`
  - `../experiment_cards/2026-06-02-haze4k-apdr-convir-v0-2rc-budget.md`
  - `../experiment_cards/2026-06-03-haze4k-apdr-v0-3-shed-diagnostics.md`
  - `../experiment_cards/2026-06-03-haze4k-apdr-v0-4-cclf-diagnostics.md`
  - `../experiment_cards/2026-06-03-haze4k-apdr-v0-4a-low-field-only.md`
  - `../experiment_cards/2026-06-03-haze4k-apdr-v0-4b-derived-lowfield-basis.md`
  - `../experiment_cards/2026-06-03-haze4k-apdr-v0-4b-mapping-triage.md`
  - `../experiment_cards/2026-06-03-haze4k-apdr-v0-4d-spatial-coeff-probe.md`
  - `../experiment_cards/2026-06-03-haze4k-apdr-v0-4e-risk-calibrated-action-bank.md`
  - `../experiment_cards/2026-06-03-haze4k-apdr-v0-4e-oof-calibration.md`
- Evidence roots are listed in `../EXPERIMENT_INDEX.md` under each APDR row.

## Established Facts

| Route group | Main evidence | Decision impact |
| --- | --- | --- |
| APDR v0/v0.1 stop20 | v0 mean `-0.00665 dB`, hard `-0.00097 dB`, easy `-0.01509 dB`, strong-reference regressions `100/250`; v0.1 mean `+0.00011 dB`, hard `+0.00067 dB`, strong-reference regressions `1/250`, severe `0/1000`. | v0 failed; v0.1 fixed preservation but did not produce hard gain. |
| v0.2/v0.2R selector-only | v0.2 AUC `0.7686` but hard/easy `H_img` ratio `1.002` and Spearman `-0.354`; v0.2R AUC `0.9766`, Spearman `-0.7466`, but easy top-25% mean `B_img` too high at `0.146`. | Hard/easy ranking improved, but selector/budget was not deployable enough for residual training. |
| v0.2RC conservative budget | Train-selected budget closed held-out easy/strong-reference mean budget to `0.002531`, retained hard mean `0.378346`, but held-out calibration BCE failed at `1.6191`. | Residual/oracle run blocked; hard-open and easy-veto need decoupling. |
| v0.4 diagnostics | Cache exactness passed; sigma `3` lowpass oracle strongest on train128; sigma `7` free-parameter low recovery `1.0938`, corr `0.9322`; train-calibrated correctability test AUC `1.0`; color branch failed. | Low-field-only candidate supported; full low+color route blocked. |
| v0.4A/v0.4B/v0.4B-MT | LowFieldNet-v1 and deployable basis/local/veil forms failed Gate B or Gate C; K32 Gate C mini-val failed with L1 drop `-0.3435`, corr `0.2154`, recovery `0.0428`, easy `-0.3551 dB`, strong/severe `11/25`; global-stat mapper rescue failed. | Local correction and stop20 blocked for these forms. |
| v0.4D/v0.4E | v0.4D same-split confidence fallback found diagnostic positives, but broad nonzero rows had strong/severe regressions; v0.4E E0 passed on confirm slice; v0.4E E1 OOF failed: Rule A keep `239/3000`, mean `+0.0749 dB`, hard `+0.2596 dB`, strong/severe `0/5`, coverage `0.0797`; Rule B keep `150/3000`, mean `+0.0378 dB`, hard `+0.1352 dB`, strong/severe `0/1`, coverage `0.0500`; policy search found `0` gate-passing policies. | No E2, full router, local correction, dense residual, or stop20 from current v0.4E. Fixed-code rerun required before numeric sealing. |

## Family Verdict

APDR produced several useful diagnostics: anchor-preserved residuals can avoid
some preservation collapse, hard/easy ranking can be learned, low-frequency
oracle targets have headroom, and safe-subset action rules can show local
positive movement. However, every deployable path so far failed either hard gain,
calibration, held-out safety, basis generalization, or OOF coverage/severe gates.

The current broad APDR output-residual and coefficient-mapping forms are stopped.
The only APDR-like future path supported by the evidence is a separately
pre-registered safe-subset route that passes fixed-code OOF or held-out gates
without severe regressions.

## Do Not Repeat Without New Evidence

- Do not launch APDR residual training from v0, v0.1, v0.2, v0.2R, or v0.2RC as-is.
- Do not continue v0.4A/v0.4B basis/local correction to stop20 from the failed
  Gate B/Gate C forms.
- Do not use the v0.4D same-split positive rows as deployment evidence; they are
  diagnostic only.
- Do not run E2, full spatial router, local correction, dense residual, or
  stop20 from current v0.4E thresholds.
- Do not seal v0.4E numeric claims until the fixed-code rerun resolves the
  documented implementation mismatch.

## Reopen Condition

A future APDR route must be pre-registered as a safe-subset or redesigned
selector route, run with fixed code, and pass OOF or fresh held-out gates for
severe regressions, strong-reference safety, coverage, hard gain, and oracle
recovery before any long training or locked test.
