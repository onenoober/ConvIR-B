# Haze4K v1.5 Full UDPNet Phase 0 Evidence

Status: `PHASE0_REPRODUCTION_GATE_FAIL`; official checkpoint eval completed on
replacement `dehaze1`; transplant, teacher distillation, and locked test are not
authorized for this checkpoint/protocol.

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
- Controlled official eval completed in `phase0_official_eval/`.
- Result: `PHASE0_REPRODUCTION_GATE_FAIL`.

## Official Eval Result

| Split | Mean delta | Hard bottom-25 | Easy top-25 | SSIM delta | Positive ratio | Strong regression ratio | Worst `<= -0.20 dB` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `val_regular` | `-0.302019 dB` | `+0.535580 dB` | `-0.796853 dB` | `-0.000345` | `0.450000` | `0.613333` | `148/300` |
| `val_hard` | `+0.426029 dB` | `+0.621163 dB` | `+0.267492 dB` | `-0.000276` | `0.610000` | `0.440000` | `104/300` |

## Decision

Do not start FullUDP transplant, teacher distillation, or locked Haze4K test
from this checkpoint/protocol. The official checkpoint provides hard-gain
diagnostic evidence, but preservation and tail safety fail the Phase 0 gate.
