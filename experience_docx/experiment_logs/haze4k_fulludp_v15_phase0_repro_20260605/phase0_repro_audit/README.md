# Haze4K v1.5 Full UDPNet Phase 0 Evidence

Status: `BLOCKED_CHECKPOINT_UNAVAILABLE`.

Primary files:

- `udpnet_convir_repro_eval.json`: official checkpoint acquisition/protocol audit.
- `udpnet_convir_bucket_compare.csv`: placeholder row because eval did not run.
- `udpnet_convir_failure_audit.csv`: blockers that prevented reproduction eval.
- `udpnet_protocol_diff.md`: checkpoint, entrypoint, data, depth, and metric diff.
- `baidupcs_transfer_probe.log`: BaiduPCS-Go public-share transfer probe.

Decision:

- Official ConvIR+UDP Haze4K checkpoint could be listed in the Baidu share but
  could not be downloaded as a plain checkpoint artifact in this environment.
- Do not claim README-level UDPNet reproduction until the official checkpoint is
  available with a recorded sha256 and a controlled ConvIR_UDPNet eval wrapper.
- Locked Haze4K test remains blocked.
