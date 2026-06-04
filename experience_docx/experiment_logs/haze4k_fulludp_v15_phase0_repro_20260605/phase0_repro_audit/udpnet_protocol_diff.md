# UDPNet ConvIR Phase 0 Protocol Diff

Status: `BLOCKED_CHECKPOINT_UNAVAILABLE`

## Checkpoint Source

- Official share: `https://pan.baidu.com/s/1JqB-YBPzZAiQsdLlNcidLQ?pwd=2026`.
- Target checkpoint: `ConvIR_UDPNet_haze4k.ckpt`.
- Listed target item: `{'server_filename': 'ConvIR_UDPNet_haze4k.ckpt', 'path': '/UDPNet_checkpoints/ConvIR_UDPNet_haze4k.ckpt', 'fs_id': 883266741305581, 'size': 108206629, 'md5': '9f085fcb3g0ddd6bfaf276e9807bbef2', 'isdir': 0}`.
- Local official checkpoint path: `/root/autodl-tmp/workspace/UDPNet_checkpoints/ConvIR_UDPNet_haze4k.ckpt`.
- Local official checkpoint available: `False`.
- A0 checkpoint: `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`.
- A0 sha256: `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`.

## Download/Access Result

- Baidu share verify/list succeeded and exposed the official checkpoint list.
- `api/sharedownload` did not expose a plain HTTP `dlink`; it returned a client
  encrypted task list for the target checkpoint.
- `BaiduPCS-Go transfer --download` did not retrieve share metadata without a
  logged-in account.
- Browser-side click invokes the BaiduNetdisk desktop client path, not a normal
  browser-download artifact that can be archived by this run.

## Repository/Entrypoint Diff

- UDPNet repo: `/root/autodl-tmp/workspace/UDPNet`.
- UDPNet commit: `f925387e690ae6016ffbd4b1cfd8490d75d7a334`.
- `ConvIR_UDPNet.py` exists: `True`.
- Official `Dehazing/ITS/test.py` imports FSNet by default:
  `True`. A ConvIR reproduction needs a
  controlled ConvIR_UDPNet build/eval wrapper rather than the unmodified test
  entrypoint.

## Data/Depth Layout Diff

- Current Haze4K root: `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`.
- Existing test hazy directory count: `1000`.
- Existing test haze directory count: `1000`.
- Official UDPNet loader expects `test/hazy`, `test/gt`, and `test/depth2l`.
- Current project data uses `haze` plus a separate DepthAnything cache:
  `/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf`.
- A future eval wrapper must symlink or adapt `haze -> hazy` and map/cache
  DepthAnything outputs into UDPNet's `depth2l` contract.

## Evaluation Status

No PSNR/SSIM evaluation was launched because the official checkpoint is absent.
This is an acquisition/protocol blocker, not a scientific failure of UDPNet.
