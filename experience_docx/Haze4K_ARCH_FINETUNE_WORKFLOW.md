# Haze4K Architecture Fine-Tune Workflow

Date: 2026-06-10

Status: dedicated workflow for fast validation of ConvIR-B Haze4K architecture
changes by partial loading the official pretrained checkpoint, freezing trusted
modules, and progressively fine-tuning only the declared new route.

## 0. Authority And Scope

This workflow is the required route pattern when a new Haze4K model structure
is proposed after the official architecture anchor was established.

Highest-priority rules:

- Default runtime server is `convir-5090`; invoke it as `ssh convir-5090`.
- Local WSL checkout is for editing and compile/static checks only. Do not run
  training, smoke tests, evaluation, inference, or demos locally.
- Treat `github/codex/haze4k-official-arch-anchor` as an immutable official
  ConvIR-B Haze4K architecture anchor.
- Every new architecture change starts from the anchor as a new branch:
  `codex/<new-route>`.
- The anchor branch itself must not be modified to host experiments.
- If a new branch reuses the official Haze4K pretrained checkpoint, it must
  write explicit partial-load and new-module initialization rules before any
  cloud run.

Current `convir-5090` anchor preflight evidence:

- Evidence root:
  `experience_docx/experiment_logs/haze4k_official_arch_anchor_convir5090_preflight_20260610/`
- Runtime workspace:
  `/home/caozhiyang/ConvIR-B/repos/ConvIR-B-official-arch-anchor`
- Python:
  `/home/caozhiyang/ConvIR-B/envs/convir-cu128/bin/python`
- Haze4K data:
  `/home/caozhiyang/ConvIR-B/datasets/Haze4K/Haze4K`
- Official Haze4K checkpoint:
  `/home/caozhiyang/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`
- Anchor commit recorded by the preflight: `2d529d4`
- Checkpoint sha256:
  `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`
- Final marker: `OFFICIAL_ANCHOR_CONVIR5090_PREFLIGHT_OK`

Use this workflow for routes such as a depth-transmission adapter, semantic
router, event/temporal residual bridge, small preservation adapter, or any
other model-structure change. Loss-only and analysis-only routes can cite the
general protocols instead, unless they also alter architecture or checkpoint
loading.

## 1. Branch And Workspace Start

Start from the immutable anchor, not from an earlier experimental leaf branch:

```bash
git fetch github '+refs/heads/*:refs/remotes/github/*'
git switch --detach github/codex/haze4k-official-arch-anchor
git switch -c codex/<new-route>
```

Required route naming:

- branch: `codex/<new-route>`;
- cloud workspace:
  `/home/caozhiyang/ConvIR-B/repos/ConvIR-B-<new-route>`;
- evidence root:
  `experience_docx/experiment_logs/<route_id>/`;
- route card:
  `experience_docx/experiment_cards/<date>-<route_id>.md`;
- model name:
  `ConvIR-Haze4K-<route-short>-<scope>-seed<seed>-<date>`.

Before writing model code, create or update the route card with:

- one-sentence hypothesis;
- exact architecture insertion points;
- new parameter prefixes;
- partial-load allowlist;
- frozen/trainable scopes;
- stage ladder and stop gates;
- locked-test policy;
- text evidence paths.

If the current working tree has unrelated changes, do not clean or revert them.
Create the route branch/workspace from a clean anchor checkout on `convir-5090`
or use a separate worktree.

## 2. Architecture Change Contract

Architecture changes should be additive and neutral by default.

Required implementation shape:

- Put new modules under clear prefixes such as `DTA_`, `SCR_`, `ETRB_`,
  `ROUTE_`, or another route-specific prefix written in the card.
- Prefer a wrapper/builder such as `build_<route>_net(...)` over editing the
  official `build_net(...)` behavior in place.
- Add an `--arch <route>` value only on the route branch.
- Keep `--arch convir --version base --fam_mode original` equivalent to the
  anchor.
- Keep all checkpoint, save, resume, padding, and three-scale output contracts
  compatible with existing Haze4K tools unless the route card explicitly
  changes them.
- Do not change tensor shapes of official pretrained layers for a fast
  fine-tune route. If shape changes are unavoidable, the affected layer is no
  longer a partial-load reuse target and must be listed as reinitialized.

Recommended file touch points on a route branch:

| File | Route-branch change |
| --- | --- |
| `Dehazing/ITS/models/<Route>ConvIR.py` | New wrapper/model modules and route-specific builder. |
| `Dehazing/ITS/main.py` | Add `--arch <route>`, builder dispatch, and explicit partial-load allowlist. |
| `Dehazing/ITS/train.py` | Add route-specific train scopes, freeze rules, and training-mode handling. |
| `Dehazing/ITS/eval.py` or tools | Only if evaluation needs route-specific prior inputs. |
| `experience_docx/tools/` | Add gate or audit scripts when existing gates do not measure the route. |

Neutral-init requirements:

- Residual correction heads should start at zero output or with residual scale
  `0.0`.
- Gates should start closed or conservative, e.g. negative bias or small
  bootstrap scale.
- FiLM-like modulation should initialize to identity:
  `gamma ~= 0`, `beta ~= 0`, applied as `x + scale * f(x, prior)`.
- Router/mask heads should have a no-op fallback and report intervention
  ratio.
- Teacher or prior encoders are frozen unless the card explicitly tests
  teacher fine-tuning.

The first preflight must prove either exact no-op equivalence to A0 or a written
bounded difference. A non-neutral branch is allowed only when the route card
states why and how the difference is controlled.

## 3. Partial-Load Rule For Haze4K Pretrained Weights

The trusted checkpoint is the official Haze4K ConvIR-B checkpoint. A route may
reuse it only through strict partial loading:

- matching official ConvIR-B keys must load exactly;
- missing keys are allowed only if they start with the route's declared new
  prefixes;
- unexpected checkpoint keys are fatal;
- shape mismatches in official keys are fatal;
- missing/unexpected/shape-mismatch keys must be printed and written to JSON;
- checkpoint path, file size, and sha256 must be recorded in the evidence
  README and route card;
- if `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is needed for a trusted legacy
  checkpoint, record it in `status.txt`.

Reference partial-load logic:

```python
def load_haze4k_partial(model, checkpoint_path, allowed_new_prefixes):
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    model_state = model.state_dict()
    loaded = {}
    shape_mismatch = []
    unexpected = []

    for key, value in state.items():
        if key not in model_state:
            unexpected.append(key)
        elif model_state[key].shape != value.shape:
            shape_mismatch.append((key, tuple(value.shape), tuple(model_state[key].shape)))
        else:
            loaded[key] = value

    missing = [key for key in model_state if key not in loaded]
    bad_missing = [
        key for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_new_prefixes)
    ]
    if unexpected or shape_mismatch or bad_missing:
        raise RuntimeError(
            f"partial-load failed: unexpected={unexpected}, "
            f"shape_mismatch={shape_mismatch}, bad_missing={bad_missing}"
        )

    model_state.update(loaded)
    model.load_state_dict(model_state, strict=True)
    return {
        "loaded": sorted(loaded),
        "missing_new_modules": sorted(missing),
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
    }
```

The route card must include a concrete allowlist, for example:

```text
allowed_new_prefixes = ("DTA_", "SCR_")
new module init:
- DTA_*: Kaiming conv weights, zero last projection, gate bias -4.0
- SCR_*: Kaiming conv weights, zero residual head, route logits bias to no-op
official ConvIR-B keys: strict shape match required
```

Do not use broad allowlists such as `missing=*`, and do not ignore unexpected
keys to make a run start.

## 4. Freeze And Progressive Fine-Tune Ladder

The default goal is fast usefulness validation, not immediate full training.
Use the smallest trainable set that can test the mechanism.

| Stage | Trainable modules | Budget | Purpose | Continue only if |
| --- | --- | --- | --- | --- |
| Stage 0 preflight | none or fixed-batch only | synthetic + one train batch | prove load, no-op, shape, finite loss, prior availability | strict partial-load passes; no-op or bounded-diff passes; no locked test touched |
| Stage 1 adapter-only | new module prefixes only | smoke/1 epoch, then up to 5 epochs | reject collapse and verify branch can learn | finite gradients, stable loss scale, cost within limits, no catastrophic val damage |
| Stage 2 adapter+neighbors | new modules plus declared nearby fusion layers | up to 20 epochs | test whether local integration helps | internal regular/hard gate passes or mechanism evidence justifies one narrow exception |
| Stage 3 selected backbone | new modules plus a small declared backbone subset | up to 80 epochs | decide whether route deserves serious training | matched quality, mechanism, preservation, and cost remain plausible |
| Stage 4 full fine-tune | explicitly declared, low LR | full budget only after Stage 3 | final candidate evidence | only after checkpoint selection and locked-test policy are fixed |

Default freeze scopes:

- `adapter_only`: train only route prefixes; freeze ConvIR-B backbone and keep
  frozen modules in eval mode.
- `adapter_neighbor`: train route prefixes plus explicitly listed adjacent
  layers, usually with a lower LR for neighbors.
- `selected_backbone`: train route prefixes plus predeclared backbone stage(s).
- `all`: allowed only after earlier gates pass or when the route is explicitly
  not a fast partial-load validation.

Default LR grouping:

```text
new modules:      1e-4 to 2e-4
neighbor layers:  1e-5
selected backbone: 5e-6 to 1e-5
weight decay:     1e-4 unless the route card explains otherwise
grad clip:        keep the existing ConvIR-B contract unless diagnosed
```

Every train log should print:

- trainable and frozen parameter counts;
- trainable prefix list;
- optimizer LR groups;
- whether frozen backbone modules are in eval mode;
- route-specific branch activity, such as gate mean, residual norm, mask
  coverage, or prior confidence.

## 5. Minimum Cloud Preflight Script

Create a durable script under the route evidence root before running it on
`convir-5090`. The script must use the new server paths and explicit Python.

Template:

```bash
#!/usr/bin/env bash
set -euo pipefail
BASE=/home/caozhiyang/ConvIR-B
WORK=$BASE/repos/ConvIR-B-<new-route>
ITS=$WORK/Dehazing/ITS
EVID=$WORK/experience_docx/experiment_logs/<route_id>
PY=$BASE/envs/convir-cu128/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
STATUS=$EVID/status.txt
LOG=$EVID/preflight_<route_id>.log
JSON_OUT=$EVID/preflight_<route_id>.json
export CUDA_VISIBLE_DEVICES=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mkdir -p "$EVID"
{
  echo "preflight_start <route_id> $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "data=$DATA"
  echo "a0=$A0"
  echo "python=$PY"
} | tee -a "$STATUS"

cd "$ITS"
set +e
PYTHONUNBUFFERED=1 "$PY" <route_preflight.py> \
  --data_dir "$DATA" \
  --checkpoint "$A0" \
  --output "$JSON_OUT" \
  > "$LOG" 2>&1
rc=$?
set -e
echo "preflight_done rc=$rc <route_id> $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "<ROUTE>_PREFLIGHT_OK" | tee -a "$STATUS"
else
  echo "<ROUTE>_PREFLIGHT_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
```

Minimum JSON fields:

- `branch`, `commit`, `python`, `torch_version`, `cuda_device_name`;
- `checkpoint`, `checkpoint_sha256`;
- `data_dir`, train/validation sample counts;
- `partial_load.loaded_count`, `missing_new_modules`, `unexpected`,
  `shape_mismatch`;
- `parameter_count_total`, `parameter_count_trainable_by_scope`;
- `synthetic_output_shapes`, `synthetic_forward_finite`;
- `noop_or_bounded_diff_vs_a0`;
- `one_batch_forward_finite`, `one_batch_loss`;
- `locked_test_touched=false`;
- `pass=true/false`.

## 6. Stage 1/2 Training Command Pattern

Training must run on `convir-5090` and should follow the existing Haze4K
workflow: durable script, status markers, stdout/stderr log, unique model name,
and no overwrite of existing outputs.

Adapter-only template:

```bash
PYTHONUNBUFFERED=1 "$PY" main.py \
  --model_name "$MODEL_NAME" \
  --data Haze4K \
  --version base \
  --fam_mode original \
  --arch <route_arch> \
  --<route>_train_scope adapter_only \
  --mode train \
  --data_dir "$DATA" \
  --batch_size 8 \
  --leaning_rate 0.0001 \
  --weight_decay 0.0001 \
  --grad_clip_norm 0.001 \
  --num_epoch 1000 \
  --stop_epoch 5 \
  --print_freq 50 \
  --num_worker 8 \
  --save_freq 5 \
  --valid_freq 1 \
  --init_model "$A0" \
  --seed 3407 \
  > "$TRAIN_LOG" 2>&1
```

Stage 2 should change only the declared scope and LR grouping, for example:

```text
--<route>_train_scope adapter_neighbor
--<route>_neighbor_learning_rate 0.00001
--stop_epoch 20
```

Do not silently change data split, active modules, loss weights, seed, or
checkpoint after seeing Stage 1 results. A changed scope is a new declared
stage or a new route id.

## 7. Internal Evaluation And Gates

No locked Haze4K test should be used to select checkpoint, scope, scale,
threshold, or active module.

Default internal evaluation:

- compare A0 vs candidate Best and Final;
- evaluate at least `val_regular` and `val_hard` when split JSON exists;
- use the same data decoding, padding, and metric code as existing tools;
- write compare JSON and per-image CSV;
- write one gate JSON with pass/fail and next action.

Minimum metrics:

- mean and median PSNR delta;
- SSIM delta when available;
- hard bottom-25% PSNR delta;
- easy/top-reference preservation delta;
- positive ratio;
- strong-reference regression ratio and count;
- worst regression count at `<= -0.20 dB`;
- route-specific activity, e.g. gate coverage, residual norm, confidence, or
  prior consistency;
- parameter, latency, and peak memory deltas when measured.

Fast-validation default gates:

| Gate | Continue line |
| --- | --- |
| Preflight | strict partial-load, finite forward, no-op/bounded-diff, no locked test |
| 5-epoch adapter-only | no collapse; regular mean not worse than A0 by more than the card limit; branch activity nonzero but bounded |
| 20-epoch adapter+neighbor | hard/target group improves or mechanism metric strongly moves; strong-reference and worst-tail budgets remain within the card |
| 80-epoch selected-backbone | matched quality, mechanism, preservation, and cost are all plausible enough for full training or a clean locked confirmation |

For current Haze4K single-seed work, remember the existing stop20 noise floor:
mean PSNR std `0.2206 dB` and hard-bucket std `0.4551 dB`. A single-seed gain
below `+0.10 dB` is a directional/mechanism signal, not promotion evidence.

## 8. Evidence And Closeout

Every stage must leave text evidence under `experience_docx/experiment_logs/<route_id>/`:

- route README with status and key metrics;
- command scripts used;
- `status.txt` with start/done markers;
- train/eval/preflight logs;
- partial-load JSON;
- compare JSON/CSV;
- gate JSON;
- route-specific mechanism audit;
- decision label and next action.

After any cloud run completes, sync text evidence back to GitHub before calling
the run closed:

- update the route card;
- update `EXPERIMENT_INDEX.md`;
- update the relevant `family_summaries/` file if a family verdict changes;
- update the evidence README;
- commit and push only text evidence and small structured artifacts;
- do not commit checkpoints, model weights, datasets, images, arrays, archives,
  or raw inference outputs.

If GitHub push is unavailable, report the exact local evidence paths and the
reason the push failed.

## 9. Route Card Checklist

Before launching Stage 0, the card must answer:

- What failure mode from existing evidence is targeted?
- Why is this route materially different from failed prior routes?
- Which anchor branch and commit is used?
- Which checkpoint path and sha256 is used?
- Which new parameter prefixes are allowed to be missing?
- Which new modules are initialized to zero, identity, or conservative gates?
- Which modules are frozen in Stage 1?
- Which modules may unfreeze in Stage 2 and at what LR?
- Which internal splits and gates decide continuation?
- Which route-specific mechanism metric proves the branch is active?
- What result stops the route without more training?
- When, if ever, is locked Haze4K test allowed?

## 10. Do-Not-Do List

- Do not edit or force-push `github/codex/haze4k-official-arch-anchor`.
- Do not branch a new architecture route from a failed experimental leaf unless
  the route card states why anchor parity is not required.
- Do not run runtime validation locally in WSL.
- Do not use broad `strict=False` loading without checking every missing and
  unexpected key.
- Do not change official pretrained layer shapes and still claim clean
  Haze4K pretrained reuse.
- Do not train the whole backbone first when adapter-only can test the
  mechanism.
- Do not tune thresholds, scales, active modules, or checkpoints from locked
  Haze4K test results.
- Do not overwrite an existing output directory or tmux session.
- Do not call a route positive from mean PSNR alone; preservation, mechanism,
  and cost evidence are required.
