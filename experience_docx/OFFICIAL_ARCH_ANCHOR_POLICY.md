# Official Architecture Anchor Policy

Date: 2026-07-16

`github/codex/haze4k-official-arch-anchor` is immutable. Any change to model
behavior, runtime entrypoint, data policy, loss, selector, gate, adapter or
evaluation behavior starts from its exact GitHub commit in a fresh
`codex/<route>` branch and fresh local/cloud workspaces. Never modify or
force-push the anchor.

Before runtime, the route card records the anchor commit, checkpoint path/hash,
strict or partial-load contract, new-module prefixes and initialization,
trainable scope, data/split/metric identities, locked-test policy, runner, cloud
paths and compact evidence paths. Local WSL remains syntax/compile-only.

The anchor may receive only documentation, command-reliability or compact
anchor-evidence maintenance that cannot alter protected model/runtime behavior.
Any protected compatibility change uses a separate maintenance branch and
explicit review.

Architecture-specific loading/freezing rules live in
`Haze4K_ARCH_FINETUNE_WORKFLOW.md`; universal design/runtime/archive rules are
not repeated here.
