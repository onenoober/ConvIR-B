# Haze4K v1.5 Full UDPNet Phase 0 Evidence

Status: `BLOCKED_CHECKPOINT_UNAVAILABLE`; Phase 0 audit completed on
`2026-06-05T01:33:42+08:00` on `dehaze1`.

This evidence root is for `ConvIR-Dehaze-v1.5-FullUDP`, Phase 0 official
UDPNet ConvIR reproduction audit.

Primary files:

- `run_v15_phase0_repro_audit.sh`: cloud audit launcher.
- `phase0_repro_audit/udpnet_convir_repro_eval.json`: checkpoint/protocol/eval
  summary.
- `phase0_repro_audit/udpnet_convir_bucket_compare.csv`: A0 vs official UDPNet
  bucket compare when the checkpoint is available.
- `phase0_repro_audit/udpnet_convir_failure_audit.csv`: blocker/failure audit.
- `phase0_repro_audit/udpnet_protocol_diff.md`: Haze4K, depth, metric,
  checkpoint, and entrypoint differences.
- `phase0_repro_audit/baidupcs_transfer_probe.log`: BaiduPCS-Go public-share
  transfer probe.
- `status.txt`: cloud run status markers.

Locked Haze4K test is not part of Phase 0 unless a fixed checkpoint and
internal reproduction gate are recorded first.

## Result

- Official share URL: `https://pan.baidu.com/s/1JqB-YBPzZAiQsdLlNcidLQ?pwd=2026`.
- Target checkpoint listed: `ConvIR_UDPNet_haze4k.ckpt`, size `108206629`
  bytes, `fs_id=883266741305581`.
- Local official checkpoint path checked:
  `/root/autodl-tmp/workspace/UDPNet_checkpoints/ConvIR_UDPNet_haze4k.ckpt`.
- Local official checkpoint available: `false`.
- Blocking evidence:
  - `official_checkpoint_missing`;
  - `baidu_share_no_plain_dlink`;
  - `baidupcs_public_transfer_failed`.
- `udpnet_convir_repro_eval.json` records
  `status=BLOCKED_CHECKPOINT_UNAVAILABLE` and no PSNR/SSIM eval was run.

## Decision

Treat official UDPNet as an architecture reference until the checkpoint is
available with sha256 and a controlled ConvIR_UDPNet eval wrapper. Do not start
FullUDP transplant or teacher distillation from README-level claims alone.
