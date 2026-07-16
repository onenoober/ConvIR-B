# Experiment Control v4 Validation

Date: 2026-07-16

Status: `PASS`

## Candidate

- GitHub branch: `codex/simplify-experiment-control-v4-20260716`
- Candidate commit: `03eac742e61f290c767086714a4792491cc69bbe`
- Runtime host: `convir-4090`
- Cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`
- Validator: `experience_docx/tools/validate_experiment_control.sh`
- Cloud root: `/tmp/convir-control-v4-03eac742e61f`

## Result

```text
state=PASS
candidate=03eac742e61f290c767086714a4792491cc69bbe
model_calls=0
tests=21
tools=6
CONTROL_V4_CONTRACT_OK tools=6 model_calls=0
CONTROL_V4_OK
```

The suite checked the schema-v4 manifest, exact card/runner/rule identities,
later-stage typed-closeout authorization, null-decision engineering closeouts,
single-fetch planning, signed state, bounded monitoring, terminal receipt
closure, evidence allowlists, the generic route-card validator, and exactly six
model-visible tools. It launched no experiment, model, training, evaluation,
inference, or GPU runner.

## Finite Repair Record

The first validation transport timed out before tests while cloning the full
repository. One inspection showed only Git clone/index-pack processes and no
status file. The orphan `/tmp` clone was terminated and removed. The single
permitted root-cause repair changed the validator to a shared clone from an
existing cloud Git object store plus one exact candidate-ref fetch. The
corrected validation then passed 21 tests.
