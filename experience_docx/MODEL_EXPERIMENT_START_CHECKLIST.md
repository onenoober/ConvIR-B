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

Follow `ROUTE_READY_FASTPATH.md`. Create only: one route card, one
`experience_docx/route_operations.json`, one runtime spec per listed operation,
one Python entrypoint, one evidence-directory README, and a compact typed asset
manifest only when needed. Every operation references the unchanged generic
runner from GitHub main. Do not create route-specific shell runners,
dispatcher/model-task files, initial-authorization JSON, validators, validator
selfproof, repair closeouts, path wrappers, or duplicate current-state
documents.

The card names the first operation. Every later operation requires the exact
prior typed closeout. Semantic-preserving engineering repair keeps the same
scientific contract and changes only the necessary code/output identity.

## Static Gate

Check source/runner/entrypoint identities; data, split, checkpoint and asset
contracts; load/init/freeze/recovery; input whitelist, no-op, shapes and finite
behavior that can be checked without local runtime; metrics/gates; output,
heartbeat, closeout and retention. Stage the complete bundle and run the one
route-ready validator. It includes the existing launch-ready card validator and
the exact MCP parser. Repeat only after a relevant contract change.

```text
git add <complete-route-bundle>
python3 experience_docx/tools/validate_route_ready.py \
  --repo . --operation <OPERATION_ID> --report /tmp/route-ready.json
```

A route is `PLANNED` only after `ROUTE_READY_OK`, when the bundle is committed
on its route branch, the first stage is the cheapest decisive stage,
runtime/resource checks are deferred to MCP plus the generic contract phase,
and no adaptive branch or placeholder is open. Do not repeat manually any
identity/path/resource check already owned by those layers.
