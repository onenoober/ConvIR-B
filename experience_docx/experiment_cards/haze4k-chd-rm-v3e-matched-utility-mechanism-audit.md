# Haze4K CHD-RM v3e Matched Utility Mechanism Audit

Date: 2026-07-10

Branch:
`codex/haze4k-v5-v3e-matched-utility-mechanism-audit`

Status:
`COMPLETED_GATE_PASS`

Decision:
`V3E_OPERATOR_CORRECTABILITY_MISMATCH_PRIMARY_HARD_GATE_SAFETY_TRADEOFF_SECONDARY_NO_RARM_EXPANSION`

## Objective

Determine whether v3d's bottleneck is primarily hard-gate restriction,
gated-training dynamics, operator-specific correctability mismatch, or training
contract/objective mismatch.

## Parent Evidence

v3d paused because D7c-gated FAM2 adapter-only was tail-safer than ungated
matched control but did not beat it on mean PSNR.

## Authorized Work

Run v3e-A/B/C/D internal audits only. No training, no new checkpoint-producing
run, no locked test, and no v4 expansion are authorized.

## Closeout

v3e found that D7c remains useful as a safety/actionability prior, but its
score is near-random for current FAM2 positive operator gain. The current hard
gate reduces tail risk but also drops mean utility; opening the gate can improve
mean but makes the tail unsafe.

The next route should be a no-training design/audit for a D7c safety veto plus
FAM2 operator-correctability ranker. Continuing v3d, moving to 20 epochs, or
expanding RARM/v4 is not authorized.
