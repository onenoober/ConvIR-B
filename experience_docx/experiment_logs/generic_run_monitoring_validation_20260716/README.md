# Generic Run Monitoring Validation Evidence

Status: `COMPLETED_ADOPTED`

This control-only route validates metadata-only telemetry and bounded closeout
monitoring on `convir-4090`. It uses no GPU, model, checkpoint, dataset,
inference, training, evaluation, canary, or locked test.

Expected compact files after the single operation are:

- `generic_run_monitoring_validation_summary.json`
- `generic_run_monitoring_validation_closeout.json`

The `r1` start boundary was never crossed. The committed one-time inspection
record is `prelaunch_unknown_state_inspection.txt`; validation proceeds only
under the new `generic-monitor-validation-r2` identity after the bounded
transport repair passes static review.

## Candidate Cloud Validation

`candidate-7b1ee1580-r1` ran on `convir-4090` using CPU and synthetic temporary
files only. It closed `FAILED_ENGINEERING` at the sixth telemetry test because
the validation harness used a raw substring check and matched explanatory text
containing `signal.`. The compact closeout and failure summary are archived
here. Raw `launch.log` and `runtime.log` remain cloud-only.

The failure did not authorize adoption. One deterministic semantic-audit repair
may be validated under a fresh candidate commit/output identity.

`candidate-4bbd03154-r2` then tested the semantic audit and also closed
`FAILED_ENGINEERING`. Its method-name extractor did not reach
`read_text` when the receiver was the compound
`Path("/proc") / str(pid) / "stat"` expression, so it incorrectly reported
that the required read was absent. Five of six telemetry tests passed, and its
closeout again records no model or protected-resource use. One deterministic
compound-receiver matcher correction may be tested under a new identity; all
other scope remains frozen.

`candidate-33a772268-r3` passed the full isolated cloud candidate gate: six
telemetry tests, 22 restricted control-plane tests, semantic audit, atomic/file
gate, and CPU-cost gate. Its summary and closeout are archived here with exact
SHA-256 identities. Raw logs remain at
`/sda/home/wangyuxin/ConvIR-B/runs/generic_run_monitoring_validation_20260716/candidate-33a772268-r3`.
The pass authorizes integration review only; registered-service end-to-end
validation is still required before adoption.

`candidate-48203394e-r4` strengthened that result by adding a direct unlimited-
sidecar lifecycle case. It proves the sidecar exits after its exact parent is
naturally reaped, without sending a signal. All seven telemetry and 22 control-
plane tests passed. The stronger summary, closeout, and outcome are archived
here; the adoption boundary is unchanged.

## Receipt-Bound Registered-Service E2E

The final `generic-monitor-e2e-r1` operation passed through the registered
`convir-ops 4.1.0` executable with exactly six schema-v4 tools. It completed
plan, launch, one bounded finish, closeout validation, evidence listing/fetch,
and read-only Git audit. Evidence fetch performed no Git mutation. The terminal
tuple authorizes `GENERIC_RUN_MONITORING_ADOPTION` only; it does not authorize a
model experiment.

- closeout: `generic_run_monitoring_e2e_closeout.json`, SHA-256
  `2e6f627eb5b3a6b0d3645aa0cfd91b43992d2625a8df41d345a77270cbad8f72`;
- summary: `generic_run_monitoring_e2e_summary.json`, SHA-256
  `5199e564a6da93a77ffcd8178c6415ccfb4511ae299cd79d6430087959eddb2c`;
- control report: `generic_run_monitoring_e2e_outcome.json`;
- raw runtime log remains cloud-only under
  `/sda/home/wangyuxin/ConvIR-B/runs/generic_run_monitoring_validation_20260716/generic-monitor-e2e-r1`.
