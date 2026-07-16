# Experiment Dispatcher Retirement Incident

Date: 2026-07-16

Status: archive-only terminal incident record; no current execution authority.

The A1X route started 95 external model children before any A1X cloud
workspace, run root, process, or result existed. Fifty-seven children failed;
the calls consumed 75,720,394 input tokens, including about 6.39 million
uncached tokens, 1,073,347 output tokens, and about 7 hours 51 minutes of child
wall time. Most work was repeated context loading, authorization, validator
self-proof, path repair, and transport recovery.

Terminal decision: the experiment dispatcher, request schema, launcher, tests,
and route integration are retired. Current experiment work keeps one qualified
model for the full warm task. Command and implementation failures stay
engineering state and never create another model task or scientific
authorization.

Historical Git commits preserve the deleted implementation if forensic
reconstruction is ever required. Do not restore it into the active rule, Skill,
route-card, MCP, or recovery surfaces.
