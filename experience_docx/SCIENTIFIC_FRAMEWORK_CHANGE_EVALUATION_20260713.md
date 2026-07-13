# Scientific Framework Change Evaluation

Date: 2026-07-13

Status: audit note for the 2026-07-13 scientific-design rewrite; not an
execution protocol.

## Scope And Authority

This note records why the framework changed and what risks remain. It does not
duplicate the operative rules. Current design and gate rules remain in
`ROUTE_DESIGN_FRAMEWORK.md` and `EXPERIMENT_GOVERNANCE_PROTOCOL.md`; route
setup, runtime retention, and card fields remain in their matching checklist,
operations protocol, and template.

The evaluation was performed against GitHub `main` commit
`6f91123802d9c6e9f4b46b84355ba8b1dab1371f`. No route verdict or historical
experiment evidence was changed.

## Finding

The previous framework was operationally disciplined but scientifically too
linear. Its "one primary variable" default reduced confounding only when
interactions were negligible. For routes involving optimizers, sampling,
representations, losses, selectors, executors, or policies, it could instead
hide interactions, spend multiple runs on order-dependent ablations, and leave
a failed combined route unable to distinguish absent signal from failed
learning or execution.

The framework also lacked explicit contracts for five common validity risks:

- screening a candidate and confirming it on the same evidence;
- treating a privileged oracle as proof of deployability;
- assigning predecessor failures to a newly added intervention;
- treating an underpowered non-significant result as evidence of no meaningful
  effect;
- discarding learned state needed to audit optimizer or mechanism behavior.

## Retrospective Usability Audit

The new design was replayed against the terminal CHD-RM sequence recorded on
GitHub `main@6f91123802d9c6e9f4b46b84355ba8b1dab1371f`. The mapping is direct:

| Historical route | New-framework role | What the replay shows |
| --- | --- | --- |
| v3r signed-margin operator repair | privileged feasibility oracle | establishes that direction repair has attainable headroom without claiming deployability |
| v3t zero-lock context diagnostic | `full_factorial` development screening | separates input-context and objective-family explanations in one paired 2x2 design |
| v3x projected safety constraint | `development_screening` mechanism gate | identifies a locally feasible projected update but authorizes no candidate or locked test |
| v3y cross-sample safety | `confirmation` on a frozen head with a disjoint held-out scope | tests whether the v3x mechanism transfers beyond update samples |
| v3z sealed confirmation | `sealed_final` for the exact frozen projected-head contract | closes that contract after held-out safety fails while preserving the broader scientific distinction |

This replay supports practical adoption: the revised framework represents the
actual decision sequence more precisely than a one-variable linear ladder. It
would additionally require explicit seed/split precision, inherited versus
intervention-added harm, and learned-state identity. Those additions strengthen
future evidence but do not retroactively change the v3x/v3y/v3z verdicts.

## Changes And Rationale

| Earlier default or gap | Revised approach | Scientific benefit |
| --- | --- | --- |
| one changed variable per serious run | one identifiable estimand; paired ablation only when interactions are implausible | aligns design with the claim instead of a mechanical edit count |
| combined mechanisms discouraged except as one opaque variable | full/fractional factorial designs with explicit interactions and aliasing | separates contributions and exposes synergy or antagonism |
| route framed around one preferred mechanism | null plus competing causal explanations and a discriminating observation | makes negative and mixed outcomes informative |
| expensive learnability work could precede attainable-headroom checks | privileged feasibility oracle, explicitly non-deployable | stops routes whose useful action does not exist before training a proxy |
| screening, confirmation, and locked test were not a complete data-role ledger | engineering, development, confirmation, and sealed-final roles | prevents discovery evidence from being presented as independent proof |
| one group split could dominate a generalization verdict | group-respecting repeated splits or leave-one-group-out when supported, plus seed/split uncertainty | tests whether the conclusion survives natural grouping and stochasticity |
| sample count alone stood in for decision precision | pre-result power/precision basis and equivalence/non-inferiority margins for no-effect claims | prevents underpowered null results from being reported as evidence of equivalence |
| safety summarized only at combined-system level | inherited, total, and intervention-added harm under a common anchor | attributes risk to the correct component without weakening total safety |
| learned diagnostics could retain only aggregate results | reconstructable learned-state and trace manifest in cloud `RUN_ROOT` | keeps later causal and trajectory audits possible |
| linear audit/training/policy profiles | feasibility, factorial, adaptive, and confirmatory profiles | spends budget on the uncertainty that currently blocks the route |
| generic template and execution guide carried historical CSD thresholds | remove the legacy block from current-rule files; require a route-specific matched baseline and threshold source | prevents new tasks from inheriting stale dataset-specific defaults |

## Guardrails Preserved

The rewrite does not relax repository authority, cloud-only runtime, immutable
architecture anchor, fresh-workspace, matched baseline, resource preflight,
typed gate, locked-test, or compact-evidence rules. It adds scientific
flexibility only inside a preregistered and auditable contract.

Factorial and adaptive designs are not automatically better. They are
authorized only when their estimands, factor levels or branch triggers,
pairing/randomization, multiplicity, evidence roles, and stopping behavior are
written before results. A fractional design that cannot defend its alias
assumptions cannot support promotion. An adaptive path invented after seeing
results is a new route contract, not a valid branch.

## Expected Efficiency

The main cost reduction should come from ordering questions correctly:
trustworthy state and baseline first, feasibility bound when useful, compact
interaction-aware screening next, independent confirmation for only the frozen
candidate, and sealed evidence last. This may use more cells in one screening
stage than a single ablation, but it should require fewer ambiguous route
restarts and produce higher decision value per cloud hour.

The state-retention rule deliberately avoids retaining every checkpoint. The
route card selects only the points needed by its written analysis. Raw states
remain on cloud; GitHub receives only compact identities and summaries.

## Adoption Readiness

These changes are not current project authority while they exist only in a
local worktree or review branch. They become usable defaults only after a
reviewed commit is merged to GitHub `main`, because `AGENTS.md` requires future
routes to plan from a freshly fetched `github/main` rules commit.

After that merge, every new or materially changed route must use the updated
card and pass `validate_experiment_card.py` before `PLANNED` and again before a
cloud launch. Exact resumes may keep their historical frozen contract. This
adoption rule does not retroactively relabel v3x/v3y/v3z or any other completed
route.

## Residual Risks

- Small datasets may not support both many factor cells and independent grouped
  confirmation. The route must reduce factor scope, strengthen assumptions, or
  label the result exploratory rather than manufacture independence.
- Repeated group splits can be unstable when there are very few natural groups.
  In that case use leave-one-group-out or report group-specific results and
  limit the population claim.
- Factorial designs can waste budget if every plausible factor is included.
  Competing hypotheses and the earliest decisive measurement must still screen
  what enters the design.
- Formal multiplicity or selection adjustment may reduce apparent power. That
  is a real cost of searching many candidates, not a reason to omit the search
  accounting.
- Richer route cards can become verbose. Keep only the frozen contract and
  compact decisions in Git; raw traces stay in `RUN_ROOT`.

## Verdict

Adopt the rewrite for new or materially changed routes. Existing route verdicts
remain historical facts and are not retroactively reinterpreted. Exact resumes
may retain their frozen scientific contract; any new claim, factor search,
adaptive branch, evidence-role change, or sealed-data reuse requires a new or
explicitly amended preregistered route card.
