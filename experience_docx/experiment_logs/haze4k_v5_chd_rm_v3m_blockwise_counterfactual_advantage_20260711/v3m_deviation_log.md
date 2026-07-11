# v3m Deviation Log

## 2026-07-11 A0a summary failure

The original `run_v3m_a0_common_action_oracle.sh` invocation completed all
frozen D_ref/D_rep OOF and confirm-audit replay segments. It then failed in
`percentile()` during grouped bootstrap summary construction because NumPy arrays
cannot be used as scalar booleans.

Verified retained raw outputs:

- policy rows: 43,200 rows plus header;
- OOF policy rows: 28,800 rows plus header;
- direction rows: 3,600 rows plus header;
- block rows: 7,200 rows plus header.

The recovery command is restricted to `--summarize_existing_raw`. It does not
load models or checkpoints, rerun inference, modify raw rows, change the action
set, or access locked test. The original failed launcher remains recorded in
`status.txt` and its log.

## 2026-07-11 operator-agreement compact artifact repair

The first recovered compact artifact populated operator agreement with `NaN`
because the summary reader used `mean_selected_alpha_mean`, while raw rows store
`selected_alpha_mean`. This did not affect the preregistered block16 gate. The
second recovery backs up the original JSON and agreement CSV, then rebuilds only
compact artifacts from the same verified raw rows and deterministic bootstrap
seed. It cannot run inference or change raw rows.

## 2026-07-11 compact CSV archive normalization

The shared CSV helper emitted CRLF line endings, which caused the repository's
whitespace gate to reject otherwise valid compact CSV evidence. The completed
A0a compact CSV files were normalized from CRLF to LF only; no cells, row
ordering, raw files, gate values, or JSON data changed. The v3m writer now
specifies LF for later compact outputs. The A0 source manifest retains the
pre-normalization script SHA because it records the script that produced the
scientific result.

## 2026-07-11 A0b metric-contract correction

The first A0b read-only cross-audit completed with exact fixed-alpha replay and
all mean-gap upper bounds below `0.005 dB`, but it required pointwise
dominance from the continuous-pixel analytic alpha policy. That policy solves
the unclamped residual then clamps the final prediction, while the common grid
selects among already clamped candidates. The pointwise nesting premise is
therefore false. A0b-r1 retains the frozen inputs and aggregate gap threshold,
applies `1e-6 dB` only to finite-grid numerical monotonicity, and verifies
existing p10/severe tail safety for both policies. The first result is retained
as a metric-contract mismatch and does not authorize or block A1.

## 2026-07-12 A1 smoke determinism repair

The first A1 smoke launched on the same GPU and frozen assets as A0 but omitted
the v3l-A1 random/NumPy/Torch/CUDA seed and cuDNN deterministic setup. Its
first fixed-alpha replay differed from the frozen reference, so it stopped
before processing an image; the partial cloud-only CSV has only its header.
The r1 smoke restores the exact v3l-A1 initialization and uses a separate
output root. This is an engineering failure, not an A1 observability result.

## 2026-07-12 A1 smoke fold-map repair

Smoke r1 retained the v3l-A1 deterministic initialization but built OOF folds
from only 32 names. Frozen OOF heads are keyed by folds assigned over the full
1,200-name manifest, so the subset selected the wrong fold head and still
failed fixed-alpha replay. Smoke r2 builds the full fold map first, then takes
the 32-name prefix and writes a third, separate cloud-only output root.
