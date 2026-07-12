# Haze4K Architecture Fine-Tune Workflow

Date: 2026-07-12

Status: Haze4K-specific source, partial-load, initialization, and trainable-scope
supplement for architecture routes.

## 0. Authority And Scope

Use this supplement when a new Haze4K model structure is proposed after the
official architecture anchor was established. `MODEL_EXPERIMENT_START_CHECKLIST.md`
owns one-time route setup and profile selection;
`MODEL_RUN_OPERATIONS_PROTOCOL.md` owns per-stage paths, launch, monitoring, and
closeout. This file must not define a second generic stage or sync workflow.

Highest-priority rules:

- Treat `github/codex/haze4k-official-arch-anchor` as an immutable official
  ConvIR-B Haze4K architecture anchor.
- Every new architecture change starts from the anchor as a new branch:
  `codex/<new-route>`.
- The anchor branch itself must not be modified to host experiments.
- If a new branch reuses the official Haze4K pretrained checkpoint, it must
  write explicit partial-load and new-module initialization rules before any
  cloud run.

Current `convir-4090` anchor runtime defaults:

- Runtime workspace:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor`
- Python:
  `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`
- Haze4K data:
  `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`
- Official Haze4K checkpoint:
  `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`

Historical `convir-5090` anchor preflight evidence, retained for provenance:

- Evidence root:
  `experience_docx/experiment_logs/haze4k_official_arch_anchor_convir5090_preflight_20260610/`
- Historical runtime workspace:
  `/home/caozhiyang/ConvIR-B/repos/ConvIR-B-official-arch-anchor`
- Historical Python:
  `/home/caozhiyang/ConvIR-B/envs/convir-cu128/bin/python`
- Historical Haze4K data:
  `/home/caozhiyang/ConvIR-B/datasets/Haze4K/Haze4K`
- Historical official Haze4K checkpoint:
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
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-<new-route>`;
- raw runtime root:
  `/sda/home/wangyuxin/ConvIR-B/runs/<route_id>/`;
- compact evidence staging root:
  `$REMOTE_REPO/experience_docx/experiment_logs/<route_id>/` at closeout only;
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
- selected training profile, first trainable scope, and typed stop gates;
- locked-test policy;
- text evidence paths.

If the current working tree has unrelated changes, do not clean or revert them.
Create the route branch/workspace from a clean anchor checkout on `convir-4090`
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

## 4. Trainable Scope Contract

Use the smallest trainable scope that can answer the route's declared mechanism
question. Do not force every architecture through adapter-only, neighbors,
selected backbone, and full fine-tune in sequence.

Available scope labels:

- `adapter_only`: train only route prefixes; freeze ConvIR-B backbone and keep
  frozen modules in eval mode.
- `adapter_neighbor`: train route prefixes plus explicitly listed adjacent
  layers, usually with a lower LR for neighbors.
- `selected_backbone`: train route prefixes plus predeclared backbone stage(s).
- `all`: use only when the mechanism genuinely requires global adaptation and
  the route card explains why a smaller scope cannot answer the question.

Choose the initial scope before formal results. If the mechanism requires
adjacent or backbone integration, start at that declared scope instead of
running a knowingly uninformative adapter-only stage. Widening scope is a new
stage and requires the previous typed closeout to authorize it.

There is no repository-wide learning-rate or epoch ladder. Freeze the budget,
LR groups, weight decay, and gradient policy from the matched predecessor,
baseline behavior, or a predeclared scale diagnostic. Do not tune them from the
formal result they will judge.

The raw train log should print one compact startup summary:

- trainable and frozen parameter counts;
- trainable prefix list;
- optimizer LR groups;
- whether frozen backbone modules are in eval mode;
- route-specific branch activity, such as gate mean, residual norm, mask
  coverage, or prior confidence.

## 5. Architecture Preflight Payload

Use the durable runner, `REMOTE_REPO`, `RUN_ROOT`, status markers, and typed
closeout from `MODEL_RUN_OPERATIONS_PROTOCOL.md`. This supplement adds only the
Haze4K architecture checks that the runner must invoke:

- `checkpoint`, `checkpoint_sha256`;
- `partial_load.loaded_count`, `missing_new_modules`, `unexpected`,
  `shape_mismatch`;
- `parameter_count_total`, `parameter_count_trainable_by_scope`;
- `synthetic_output_shapes`, `synthetic_forward_finite`;
- `noop_or_bounded_diff_vs_a0`;
- one real-batch finite forward; backward/loss only when the selected profile
  trains;
- `locked_test_touched=false`;
- a typed structural decision and the exact next stage it authorizes.

## 6. Training Command Contract

The tracked runner supplies the route's frozen values for `--arch`, trainable
scope, data/split, checkpoint, seed policy, optimizer groups, budget, evaluation
cadence, and output root. This supplement does not prescribe batch size,
learning rate, epoch count, save cadence, or a universal adapter-first command.

Do not silently change data split, active modules, loss weights, seed,
checkpoint, scope, or budget after seeing a formal result. A changed scope is a
new declared stage; a changed scientific comparison requires a new run id and
route-card update.

## 7. Internal Evaluation And Gates

No locked Haze4K test should be used to select checkpoint, scope, scale,
threshold, or active module.

For a formal internal evaluation:

- compare A0 with the single fixed candidate checkpoint named by the stage;
- evaluate at least `val_regular` and `val_hard` when split JSON exists;
- use the same data decoding, padding, and metric code as existing tools;
- keep raw per-image comparisons in `RUN_ROOT`;
- write one compact typed closeout with decision and next-stage authorization.

Minimum compact metrics:

- primary quality effect with uncertainty at the image/group unit;
- hard/lower-tail and strong-reference preservation summaries;
- one route-specific activity metric, such as bounded residual norm or gate
  coverage;
- parameter, latency, or peak-memory deltas only when the route's gate uses
  them.

Architecture-specific gate questions are:

| Gate type | Required question |
| --- | --- |
| structural integrity | Did strict partial-load, shapes, finite forward, neutral/bounded initialization, and locked-test protection pass? |
| scope viability | Is the declared trainable scope active and numerically healthy under its written budget? |
| scientific utility | Does matched evaluation support the mechanism while preserving strong and tail cases? |
| safety/promotion | Do direct replay, tail risk, cost, and uncertainty support only the written promotion action? |

Epoch counts are route parameters, not gate identities. The historical Haze4K
stop20 noise estimates (`0.2206 dB` mean PSNR std and `0.4551 dB` hard-bucket
std) may be cited only when the current data, metric, budget, and seed contract
match; otherwise measure or justify a current uncertainty reference.

## 8. Evidence And Closeout

Follow `MODEL_RUN_OPERATIONS_PROTOCOL.md`: raw logs, per-image tables,
checkpoints, and outputs stay in `RUN_ROOT`; only the runner, typed closeout,
compact status, evidence README, and necessary aggregate summaries enter
`EVID_STAGE`.

Commit intermediate compact evidence to the route branch. Update the central
index/family summary and sync GitHub `main` only at a terminal decision or an
explicit major handoff, as defined by `BRANCH_EXPERIMENT_SYNC_PROTOCOL.md`.

## 9. Route Card Checklist

Before launching the first cloud stage, the card must answer:

- What failure mode from existing evidence is targeted?
- Why is this route materially different from failed prior routes?
- Which anchor branch and commit is used?
- Which checkpoint path and sha256 is used?
- Which new parameter prefixes are allowed to be missing?
- Which new modules are initialized to zero, identity, or conservative gates?
- Which modules are trainable in the first informative scope?
- Which wider scope, if any, may be authorized later and what question would it
  answer?
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
- Do not train the whole backbone first when a smaller scope can answer the
  mechanism; if it cannot, state why before launch.
- Do not tune thresholds, scales, active modules, or checkpoints from locked
  Haze4K test results.
- Do not overwrite an existing output directory or tmux session.
- Do not call a route positive from mean PSNR alone; preservation, mechanism,
  and cost evidence are required.
