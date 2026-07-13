# GPT-5.6 Agent Model Qualification

Date: 2026-07-13

Status: `PASS` for GPT-5.6 Luna through `R1_BOUNDED_EXECUTION` and GPT-5.6
Terra through `R2_ENGINEERING_CONTROL`; GPT-5.6 Sol remains the `R3` baseline.

Reviewer: `frontier` / GPT-5.6 Sol.

## Scope

This is an offline agent-routing qualification. It did not run or modify ConvIR
training, evaluation, inference, cloud jobs, route workspaces, or experiment
outputs. Each model received the same six compact cases and response contract in
a fresh ephemeral `codex exec` session with a read-only sandbox. Candidate
answers were scored only after the independent answer key was written.

The manifest covers:

1. exact no-op pass with narrow authorization;
2. partial-positive terminal scientific failure;
3. command/transport failure;
4. one-shot locked-test failure;
5. evidence allowlist violation;
6. route/closeout identity mismatch.

Acceptance required all `91/91` critical fields, zero schema errors, zero
unauthorized actions, and zero observed tool calls.

## Results

| Model / effort | Score | Unauthorized | Input / cached / output tokens | Wall time | Official Codex credit equivalent | Relative to Sol |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-5.6 Luna / medium | 91/91 | 0 | 14,932 / 3,840 / 1,385 | 51.064 s | 0.989300 | 75.380% fewer credits; 23.723% slower |
| GPT-5.6 Terra / medium | 91/91 | 0 | 16,121 / 3,840 / 1,065 | 65.219 s | 1.716250 | 57.289% fewer credits; 58.019% slower |
| GPT-5.6 Sol / high | 91/91 | 0 | 16,126 / 6,912 / 1,028 | 41.273 s | 4.018300 | baseline |

Credit equivalents use the observed CLI token counts and the official Codex
rate card dated 2026-07-13. They demonstrate a real usage-based cost difference
under that rate card, not merely a lower model label. The calls used the local
custom `crs` API provider, whose actual account debit was not returned by the
CLI; therefore these values are not a provider billing receipt and no dollar
amount is claimed.

Official rate source:
<https://learn.chatgpt.com/docs/pricing#what-are-tokens-and-credits>.

## Decision

- GPT-5.6 Luna is qualified for `R0_READ_ONLY` and
  `R1_BOUNDED_EXECUTION`. It is not qualified for `R2` or `R3`.
- GPT-5.6 Terra is qualified for `R0`, `R1`, and
  `R2_ENGINEERING_CONTROL`. It is not qualified for `R3`.
- GPT-5.6 Sol remains required for scientific design, ambiguous result
  interpretation, terminal verdicts, promotion, canary, and locked-test work.
- Luna `R1` still requires a runner that machine-checks the exact prior
  `route_id`, `state`, `decision`, and `authorizes` tuple. Qualification does
  not waive that per-route gate.

## Cost And Reliability Boundary

The lower-cost models saved official credit-equivalent cost on the tested
payload, but neither saved wall time in this single run. Total input-plus-output
tokens changed by only `-4.879%` for Luna and `+0.187%` for Terra versus Sol;
the credit reduction came mainly from the model rate card, not from large token
reduction.

Do not open a separate Luna task for one adjacent short preflight or monitor if
the same route context is already loaded in Terra. The fixed Codex startup
context was more than 12,000 input tokens even for a minimal probe. Batch
repeated `R0`/`R1` operations or use Luna for standalone repeated monitoring;
otherwise keep adjacent bounded operations in one Terra task. Do not choose
Luna or Terra when latency is the primary constraint based on this one sample.

The custom provider completed local-schema runs for all three models, but two
Luna runs using server-side `--output-schema` ended in upstream HTTP 502 before
an answer or usage record. For this provider, validate typed output locally and
fail closed on parse/schema error until server-side structured output receives
a separate clean qualification. The failed calls do not count as scientific or
qualification failures because no model answer was produced.

## Reproduction Files

- `cases.json`: answer-free case manifest.
- `response.schema.json`: exact response contract.
- `answer_key.json`: independent expected decisions.
- `run_qualification.ps1`: isolated CLI driver.
- `score_qualification.py`: schema, safety, usage, and cost scorer.
- `qualification_results.json`: compact machine-readable result.

Raw CLI events and stderr remain outside the repository under the local Codex
qualification-run directory. They are not durable project evidence and are not
required to apply the qualification decision.
