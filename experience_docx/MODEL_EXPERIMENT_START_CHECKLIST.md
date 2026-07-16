# Model Experiment Start Checklist

Date: 2026-07-16

Status: one-time setup for a new route or materially changed route contract.

## 1. Ground And Freeze The Route

Record one route id and answer:

- authoritative GitHub evidence and current cloud facts;
- source branch/commit and fresh route branch/workspace;
- question, population, analysis unit, intervention/reference, outcome, and
  aggregation;
- competing hypothesis and cheapest discriminating observation;
- data-role ledger and forbidden evidence flows;
- primary gate, uncertainty unit, threshold source, and
  `PASS/INCONCLUSIVE/FAIL` meanings;
- locked-test/canary policy and terminal stop/reopen rules.

New Haze4K model-structure routes start from the immutable official anchor.
Current process rules always come from fresh GitHub `main`, not the route
checkout.

## 2. Create The Minimal Route Bundle

Create only:

1. one route card;
2. one `experience_docx/route_operations.json` machine projection;
3. one parameterized tracked runner;
4. the route implementation and, only when necessary, one separate compact
   asset manifest.

Do not create an agent-routing table, dispatcher request, initial-authorization
JSON, route-specific validator, validator selfproof, negative/positive fixture
authorization, per-repair closeout, or duplicate current-state document.

The frozen route card authorizes the first stage named by the operations
manifest. Every later stage requires the exact previous typed closeout. An
engineering repair that preserves route semantics keeps the existing
authorization and uses a new code commit/output id only when required.

## 3. Static Preflight

Verify once per relevant route commit:

- source, runner, entrypoint, dataset/split, asset and checkpoint identities;
- load/init/freeze/resume contracts;
- baseline and matched data/metric view;
- input whitelist, no-op/neutral behavior, shape/batch and finite-value checks
  that can run without experiment data locally;
- primary metrics/gates, paired/grouped uncertainty, multiplicity and data
  roles;
- output, status, closeout, retention and locked-test behavior.

Run the single generic card validator:

```text
python3 experience_docx/tools/validate_experiment_card.py <route-card>
```

It checks the compact contract shape only. Scientific correctness remains the
current R3 task's responsibility. Never add a route-specific validator to make
the route card prove its own implementation.

Rerun static validation only when a listed contract item changes. Do not rerun
because another stage starts or because a command transport failed.

## 4. Choose The Smallest Decision Sequence

Use only stages that can change a written next action. Typical forms are:

| Route type | Minimum sequence |
| --- | --- |
| audit/evaluation | integrated smoke -> formal evaluation |
| feasibility oracle | integrity check -> privileged upper bound -> stop/design review |
| training intervention | integrated smoke -> cheapest decisive scout if needed -> independent formal |
| policy/representation accessibility | engineering smoke -> frozen cross-fit/OOF confirmation |

Screening cannot prove promotion. Locked test is never debugging, fitting,
selection, or repair evidence.

## 5. Operations Manifest

Use schema v3 from `CONVIR_OPS_MCP.md`. It records a digest of the canonical
rule bundle, so unrelated GitHub `main` advancement does not invalidate a
route. Keep the manifest under 16 KiB. Put large asset records in a separate
compact manifest referenced by the runner.

For the first operation, `prior_closeout_relpath` is null and the frozen card
must name that stage. Later operations name one prior closeout and its exact
terminal tuple. Do not add an intermediate authorization file.

## 6. Ready To Launch

A route is `PLANNED` only when:

- the compact card passes the generic validator;
- source, runner, operations manifest, and required assets are committed on the
  named route branch;
- the first stage is the cheapest stage that resolves its current question;
- dynamic cloud checks are deferred to immediately before launch;
- the expected wall time, heartbeat, observation budget, output id, and
  unit-boundary resume policy are explicit;
- no unresolved placeholder or unlisted adaptive branch remains.

One root-cause implementation failure permits one repair cycle in the same
qualified task. A repeated same-class failure stops with one engineering
blocker; it does not create another authorization or task.
