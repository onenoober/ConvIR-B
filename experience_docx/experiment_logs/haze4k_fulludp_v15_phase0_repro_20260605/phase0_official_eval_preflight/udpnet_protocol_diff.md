# UDPNet ConvIR Phase 0 Official Eval Protocol Diff

Status: `PHASE0_REPRODUCTION_GATE_FAIL`

## Checkpoints

- A0 checkpoint: `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`
- A0 sha256: `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`
- Official UDPNet checkpoint: `/root/autodl-tmp/workspace/UDPNet_official_download/ConvIR_UDPNet_haze4k.ckpt`
- Official UDPNet sha256: `6d02d2a42e97cc411a36d95cfaf8421eb25a5622f0cac8c150c0e790b7149291`
- Official UDPNet metadata: `{'epoch': 1265, 'global_step': 272190, 'pytorch_lightning_version': '2.3.1', 'state_dict_key_count': 689}`

## Data And Splits

- Data root: `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`
- Depth cache: `/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf`
- Depth split: `train`
- Split JSON: `/root/autodl-tmp/workspace/ConvIR-B-v1-5-fulludp-runtime/experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604/internal_val/haze4k_dpga_v13_regular_hard_seed3407.json`
- Evaluated splits: `['val_regular']`
- Locked Haze4K test touched: `False`

This run uses the existing train-derived `val_regular` and `val_hard` split
contract before any locked-test decision.

## Entrypoint

- Local ConvIR-B A0 model is loaded from `Dehazing/ITS/models/ConvIR.py`.
- Official UDPNet model is loaded from `UDPNet/Dehazing/ITS/models/ConvIR_UDPNet.py`.
- Official `UDPNet/Dehazing/ITS/test.py` is not used because it imports FSNet by
  default and expects a `test/depth2l` directory. This wrapper evaluates
  `ConvIR_UDPNet` directly and maps the existing DepthAnything cache to the
  official 4-channel RGB+depth input.

## Metric

- PSNR: `10*log10(1/MSE) on RGB tensors in [0,1]`
- SSIM: `pytorch_msssim with adaptive average pooling matching existing ConvIR eval`
- Pad factor: `32`
