# Experiment Start Checklist

Date: 2026-07-16

Use once for a new route or a material scientific-contract change.

## Freeze

- Ground status in GitHub evidence and current cloud state.
- Record route/source commit, fresh workspaces, estimand, data-role ledger,
  locked-test policy, primary gate/uncertainty, and terminal actions.
- For Haze4K model-structure changes, start from the immutable official anchor
  and record partial-load/init/freeze rules.

## Minimal Bundle

Create only: one route card, one `experience_docx/route_operations.json`, one
parameterized runner, route code, and a compact asset manifest only when needed.
Do not create dispatcher/model-task files, initial-authorization JSON,
route-specific validators, validator selfproof, repair closeouts, or duplicate
current-state documents.

The card names the first operation. Every later operation requires the exact
prior typed closeout. Semantic-preserving engineering repair keeps the same
scientific contract and changes only the necessary code/output identity.

## Static Gate

Check source/runner/entrypoint identities; data, split, checkpoint and asset
contracts; load/init/freeze/resume; input whitelist, no-op, shapes and finite
behavior that can be checked without local runtime; metrics/gates; output,
heartbeat, closeout and retention. Run the one generic card validator. Repeat
only after a relevant contract change.

```text
python3 experience_docx/tools/validate_experiment_card.py <route-card> --launch-ready
```

A route is `PLANNED` only when the bundle is committed on its route branch, the
card validates, the first stage is the cheapest decisive stage, runtime/resource
checks are deferred to launch, and no adaptive branch or placeholder is open.
