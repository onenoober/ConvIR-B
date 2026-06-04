# Haze4K APDR-v0.4E Reproducibility Audit

Date: 2026-06-03

Status: implementation mismatch confirmed on commit `ed38afb`; fixed-code
clean rerun on `autodl-dehaze4` is required before v0.4E numerical evidence can
be sealed.

## Finding

Commit `ed38afbb60899b1045f091cc3be552a3295e995f` contained two runtime
reproducibility defects in the v0.4E action-bank audit tools:

- `F.interpolate(..., align_coners=False)` used a misspelled PyTorch keyword.
- The v0.4E scripts parsed `--kenel_size` and used `args.kenel_size`, while
  the shared `frozen_apdr_tensors()` helper requires `args.kernel_size`.

The logged E0/E1 failure direction remains useful diagnostic evidence, but the
exact numeric rows from `ed38afb` are not sealed as clean-reproducible evidence.
Do not launch E2, full router, local correction, dense residual, or stop20 from
the pre-rerun v0.4E numbers.

## Local Audit Snapshot

```text
$ git rev-parse HEAD
ed38afbb60899b1045f091cc3be552a3295e995f

$ git diff --stat
 .../run_apdr_v0_4e_oof_calibration_sigma3.sh                        | 2 +-
 .../run_apdr_v0_4e_risk_action_bank_sigma3.sh                       | 2 +-
 experience_docx/tools/audit_haze4k_apdr_v0_4e_oof_calibration.py    | 6 +++---
 experience_docx/tools/audit_haze4k_apdr_v0_4e_risk_action_bank.py   | 6 +++---
 4 files changed, 8 insertions(+), 8 deletions(-)

$ python3 -V
Python 3.10.12

$ python3 -c "import torch, inspect; import torch.nn.functional as F; print(torch.__version__); print(inspect.signature(F.interpolate))"
ModuleNotFoundError: No module named 'torch'
```

Local WSL cannot substitute for the torch signature check because torch is not
installed there. The torch signature check must be run in the clean AutoDL
environment before rerun.

Post-fix static checks:

```text
$ grep -R -n align_coners experience_docx/tools/audit_haze4k_apdr_v0_4e_*.py
<no matches>

$ grep -R -n kenel_size experience_docx/tools/audit_haze4k_apdr_v0_4e_*.py \
  experience_docx/experiment_logs/haze4k_apdr_v0_4e_*_20260603/run_apdr_v0_4e_*_sigma3.sh
experience_docx/tools/audit_haze4k_apdr_v0_4e_risk_action_bank.py:500:    parser.add_argument("--kernel_size", "--kenel_size", dest="kernel_size", type=int, default=31)
experience_docx/tools/audit_haze4k_apdr_v0_4e_oof_calibration.py:325:    parser.add_argument("--kernel_size", "--kenel_size", dest="kernel_size", type=int, default=31)
```

The remaining `--kenel_size` strings are backward-compatible argparse aliases.
New run scripts use `--kernel_size`.

Tool hashes after local fix:

```text
2d9a2211108a31355c0793e660c56f1db7d46c1bab54a00ef534c9b560c6d147  experience_docx/tools/audit_haze4k_apdr_v0_4e_risk_action_bank.py
37e709e2d7c8dd2c59f857a84cc79843042b6481c9529269c180a8be4e697a6e  experience_docx/tools/audit_haze4k_apdr_v0_4e_oof_calibration.py
```

## Required Clean Rerun

Run only on `autodl-dehaze4` from a clean checkout of the fixed commit:

```bash
git rev-parse HEAD
git diff --stat
python -V
python - <<'PY'
import torch
import inspect
import torch.nn.functional as F
print(torch.__version__)
print(inspect.signature(F.interpolate))
PY
grep -R "align_coners\|align_corners\|kenel_size\|kernel_size" -n experience_docx/tools/*.py
sha256sum experience_docx/tools/audit_haze4k_apdr_v0_4e_*.py
```

Then rerun:

```bash
bash experience_docx/experiment_logs/haze4k_apdr_v0_4e_risk_action_bank_20260603/run_apdr_v0_4e_risk_action_bank_sigma3.sh
bash experience_docx/experiment_logs/haze4k_apdr_v0_4e_oof_calibration_20260603/run_apdr_v0_4e_oof_calibration_sigma3.sh
bash experience_docx/experiment_logs/haze4k_apdr_v0_4e_oof_calibration_20260603/run_v04e_oof_policy_search.sh
```

Decision rule:

- If fixed-code E1 still fails, formally close current v0.4E thresholds.
- If fixed-code numbers change materially, mark `ed38afb` v0.4E evidence as
  `implementation-mismatch invalidated`.

## AutoDL Follow-Up

An AutoDL rerun from clean `826caaf` was completed under:

- `haze4k_apdr_v0_4e_repro_audit_20260603_autodl/`
- `haze4k_apdr_v0_4e_risk_action_bank_rerun_20260603_autodl_826caaf/`
- `haze4k_apdr_v0_4e_oof_calibration_rerun_20260603_autodl_826caaf/`

The rerun confirmed the stop direction but exposed two more implementation
issues:

- Variable-schema summary rows in E1 required union-field CSV writing.
- Historical `*_kenel_knn_9` mapper names did not match generated
  `*_kernel_knn_9` mapper names, filtering out KNN candidates in clean
  `826caaf`.

Current code now patches both issues and adds a finalize tool that can recover
E1 summary tables from the per-image intermediate CSV. Exact full v0.4E numeric
sealing still requires an alias-corrected OOF rerun; no E2 or training route is
authorized before that.
