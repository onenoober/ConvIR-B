# Experiment Control v3 Validation

Date: 2026-07-16

Status: `PASS`

## Validated Candidate

- GitHub branch: `codex/simplify-experiment-control-20260716`
- Candidate commit: `226bc4aa98d825c5b771085d514943c27e60db30`
- Runtime host: `convir-4090`
- Cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`
- Runner: `experience_docx/tools/validate_experiment_control_v3.sh`
- Cloud run root: `/tmp/convir-control-v3-226bc4aa98d8`

## Result

```text
state=PASS
candidate=226bc4aa98d825c5b771085d514943c27e60db30
model_calls=0
tests=14
tools=6
CONTROL_V3_CONTRACT_OK tools=6 model_calls=0
CONTROL_V3_OK
```

The cloud test suite completed `14/14` tests and verified MCP schema v3, server
version `3.0.0`, exactly six model-visible tools, and monitor windows no longer
than 60 seconds. The validation made no model calls and launched no experiment,
training, evaluation, or inference process.

## Finite Repair Record

The first candidate, `ac0657cf46a4323c8cbe2c7c45f1954129799530`, failed one
of 14 tests because the `standard` and `long` profiles encoded a 75-second
window. Commit `226bc4aa98d825c5b771085d514943c27e60db30` reduced those
profiles to 60 seconds and the `short` profile to its documented 30 seconds.
The corrected candidate then passed the single permitted root-cause repair
cycle. The initial failure was engineering-only and produced no scientific
evidence or new authorization.
