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

No A0R result exists yet. A cloud launch requires fresh source snapshots,
asset-hash preflight, a clean route checkout, and the route-card contract hash
recorded in the launch transcript.
