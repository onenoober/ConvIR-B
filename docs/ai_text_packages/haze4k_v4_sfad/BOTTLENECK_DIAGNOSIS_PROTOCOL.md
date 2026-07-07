# Haze4K v4.4 Bottleneck Diagnosis Protocol

Date: 2026-07-08

Status: audit-only protocol.

## Purpose

Diagnose whether A3 failed because of 1/2-scale intervention collision, SDFM field compression, GST sign flip, or frequency-band side effects.

## Inputs

- A0 official checkpoint.
- A1 SDFM-only Final checkpoint.
- A2 GST-only Final checkpoint.
- A3 SDFM+GST Final checkpoint.
- Haze4K `train/haze` and `train/gt` only.

Haze4K `test` is forbidden.

## Required Outputs

- `joint_delta_matrix.csv`
- `joint_delta_matrix_trainfit128.csv`
- `module_interaction_stats.csv`
- `correlation_report.json`
- `band_error_report.csv`
- `scale_collision_report.md`
- `failure_atlas_after_a3.md`
- `decision_after_diagnosis.md`

## Metric Contract

The primary diagnostic split is `internal_holdout256`, sampled deterministically from Haze4K train with seed `3407`. The legacy `trainfit128` split is retained only for comparability with A1/A2/A3 route evidence.

No result in this route is a generalization claim; it is train-derived internal evidence for route selection.
