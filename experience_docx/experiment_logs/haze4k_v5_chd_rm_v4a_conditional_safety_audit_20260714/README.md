# v4a Conditional Safety Audit Evidence

Status: `PLANNED`.

This route is an instrumented reconstruction and failure-identification audit
of the closed v3z projected-head contract. It is not a candidate model route,
does not train a new architecture or policy, and cannot access the Haze4K
locked test.

The exact scientific contract is in:

`experience_docx/experiment_cards/2026-07-14-haze4k-v5-chd-rm-v4a-conditional-safety-audit.md`.

The A0R runner imports the immutable v3z source at
`3caddcc5265732e5be77e3404119a28cb28c11e6`, retains raw learned states only
under the cloud run root, and stages only compact manifests, summaries, and
typed closeouts here.

No A0R scientific result exists yet. The first A0R launch completed the exact
no-op check, then stopped with `FAILED_COMMAND_OR_INFRA` before projected
reconstruction because the wrapper called a v3p legacy module as if it were the
v3w module. The failed output remains cloud-only under `RUN_ROOT/a0r`; no
projected epoch, per-image row, or gate decision was produced. The corrected
runner uses a new output id under the unchanged contract.
