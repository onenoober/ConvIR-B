# Haze4K v1.5 Full UDPNet Phase 0 Evidence

Status: Phase 0 checkpoint-acquisition blocker reopened after the official
checkpoint became available on the replacement `dehaze1`; controlled official
eval is pending.

This evidence root is for `ConvIR-Dehaze-v1.5-FullUDP`, Phase 0 official
UDPNet ConvIR reproduction audit.

Primary files:

- `run_v15_phase0_repro_audit.sh`: cloud audit launcher.
- `run_v15_phase0_official_eval.sh`: controlled official ConvIR+UDP eval on
  train-derived `val_regular` and `val_hard`; does not touch locked Haze4K
  test.
- `phase0_official_eval/udpnet_convir_repro_eval.json`: official checkpoint
  reproduction summaries and gate once the eval completes.
- `phase0_official_eval/udpnet_convir_bucket_compare.csv`: per-image A0 vs
  official UDPNet bucket comparison.
- `phase0_official_eval/udpnet_convir_failure_audit.csv`: strong/worst
  regression audit.
- `phase0_repro_audit/udpnet_convir_repro_eval.json`: checkpoint/protocol/eval
  summary from the first blocked acquisition audit.
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

Initial blocked result:

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

Reopen evidence:

- Replacement `dehaze1`: `root@connect.bjb1.seetacloud.com:42371`.
- Available official checkpoint:
  `/root/autodl-tmp/workspace/UDPNet_official_download/ConvIR_UDPNet_haze4k.ckpt`.
- sha256:
  `6d02d2a42e97cc411a36d95cfaf8421eb25a5622f0cac8c150c0e790b7149291`.
- Next action: run `run_v15_phase0_official_eval.sh` before any FullUDP
  transplant or teacher distillation.

## Decision

Treat official UDPNet as an architecture reference until the checkpoint is
available with sha256 and a controlled ConvIR_UDPNet eval wrapper. Do not start
FullUDP transplant or teacher distillation from README-level claims alone.
