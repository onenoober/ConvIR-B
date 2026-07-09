# CHD-RM v2g Need Actionability Audit Route Decision

Decision: `START_V2G_NEED_ACTIONABILITY_AUDIT_ONLY`

## Fact Sources

- GitHub main source of truth: `experience_docx/CHD_RM_EXPERIMENT_INDEX.md`.
- Prior compact evidence: v2d/v2e/v2f/F4b route cards and evidence roots on GitHub main.
- Runtime/raw-output source: cloud `convir-4090` under the v2f workspace and evidence roots listed in `v2g_source_of_truth_manifest.json`.

## Route Identity

v2g is a diagnostic audit and target-semantics route. It is not F5, not a v3/RARM continuation, not D2, and not another F4/F4b strength sweep.

Core question:

```text
Which global LDHN positives are actually CHD-RM-compatible haze-actionable recovery need,
and which are post-A0 residuals that should not drive RARM?
```

## Forbidden Work

- Haze4K locked test usage.
- D2 training or inference.
- RARM connection or training.
- v3 no-op RARM audit.
- F5 controls before v2g produces a new written target/actionability decision.
- Broad canary expansion or strength sweep of the same F4/F4b family.

## Authorized Work

- G0 source-of-truth reproduction from existing compact and cloud evidence.
- G1 LDHN semantic autopsy from existing v2e/v2f/F4b cloud outputs.
- G2 available-information upper-bound audit, only on train_inner/val_inner and only if required assets are present.
- Compact text evidence generation: CSV/JSON/MD summaries only.
