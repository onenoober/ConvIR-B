import torch
import torch.nn as nn
import torch.nn.functional as F

from .ConvIR import ConvIR


class GuidedSkipTransfer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        image_channels = max(16, channels // 8)
        hidden_channels = max(32, channels // 2)

        self.image_proj = nn.Sequential(
            nn.Conv2d(3, image_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(image_channels + channels * 4, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.clean_delta = nn.Sequential(
            nn.Conv2d(image_channels + channels * 4, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )
        self.alpha = nn.Parameter(torch.zeros(1))
        self._last_stats = {}

    def forward(self, image, decoder, skip):
        if image.shape[-2:] != skip.shape[-2:]:
            image = F.interpolate(image, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        if decoder.shape[-2:] != skip.shape[-2:]:
            decoder = F.interpolate(decoder, size=skip.shape[-2:], mode='bilinear', align_corners=False)

        image_feature = self.image_proj(image)
        skip_low = self._low_pass(skip)
        decoder_low = self._low_pass(decoder)
        skip_high = skip - skip_low
        decoder_high = decoder - decoder_low
        features = torch.cat([image_feature, skip, decoder, skip_high, decoder_high], dim=1)
        gate = self.gate(features)
        clean_skip = skip + torch.tanh(self.clean_delta(features))
        out = skip + self.alpha * gate * (clean_skip - skip)
        self._last_stats = self._summarize(gate, clean_skip - skip, skip_high, decoder_high)
        return out

    @staticmethod
    def _low_pass(x):
        return F.avg_pool2d(x, kernel_size=5, stride=1, padding=2, count_include_pad=False)

    def _summarize(self, gate, delta, skip_high, decoder_high):
        gate = gate.detach()
        delta = delta.detach()
        skip_high = skip_high.detach()
        decoder_high = decoder_high.detach()
        return {
            'gate_mean': float(gate.mean().cpu()),
            'gate_std': float(gate.std(unbiased=False).cpu()),
            'gate_min': float(gate.min().cpu()),
            'gate_max': float(gate.max().cpu()),
            'gate_lt_005': float((gate < 0.05).float().mean().cpu()),
            'gate_gt_095': float((gate > 0.95).float().mean().cpu()),
            'delta_mean': float(delta.mean().cpu()),
            'delta_std': float(delta.std(unbiased=False).cpu()),
            'delta_abs_mean': float(delta.abs().mean().cpu()),
            'delta_abs_max': float(delta.abs().max().cpu()),
            'skip_high_abs_mean': float(skip_high.abs().mean().cpu()),
            'decoder_high_abs_mean': float(decoder_high.abs().mean().cpu()),
            'alpha': float(self.alpha.detach().cpu().reshape(-1)[0]),
        }

    def collect_stats(self):
        return dict(self._last_stats)


class SFADGSTConvIR(ConvIR):
    def __init__(self, version, data):
        super().__init__(version, data)
        base_channel = 32
        self.SFAD_GST1 = GuidedSkipTransfer(base_channel * 2)
        self.SFAD_GST2 = GuidedSkipTransfer(base_channel)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)

        z = self.feat_extract[3](z)
        outputs.append(z_ + x_4)

        res2 = self.SFAD_GST1(x_2, z, res2)
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)

        z = self.feat_extract[4](z)
        outputs.append(z_ + x_2)

        res1 = self.SFAD_GST2(x, z, res1)
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z + x)

        return outputs

    def collect_modulation_stats(self, x):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            self.forward(x)
        if was_training:
            self.train()
        return {
            'GST_1_2': self.SFAD_GST1.collect_stats(),
            'GST_1_1': self.SFAD_GST2.collect_stats(),
        }


def build_sfad_gst_net(version, data, fam_mode='original'):
    if fam_mode != 'original':
        raise ValueError("SFAD-GST route starts from fam_mode='original'.")
    return SFADGSTConvIR(version, data)


def load_haze4k_partial(model, checkpoint_path, allowed_new_prefixes):
    state = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        state = state['model']

    model_state = model.state_dict()
    loaded = {}
    shape_mismatch = []
    unexpected = []
    for key, value in state.items():
        if key not in model_state:
            unexpected.append(key)
        elif tuple(model_state[key].shape) != tuple(value.shape):
            shape_mismatch.append([key, list(value.shape), list(model_state[key].shape)])
        else:
            loaded[key] = value

    missing = [key for key in model_state if key not in loaded]
    bad_missing = [
        key for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_new_prefixes)
    ]
    if unexpected or shape_mismatch or bad_missing:
        raise RuntimeError(
            'partial-load failed: '
            f'unexpected={unexpected}, '
            f'shape_mismatch={shape_mismatch}, '
            f'bad_missing={bad_missing}'
        )

    model_state.update(loaded)
    model.load_state_dict(model_state, strict=True)
    return {
        'checkpoint': checkpoint_path,
        'loaded_count': len(loaded),
        'missing_new_modules': sorted(missing),
        'unexpected': unexpected,
        'shape_mismatch': shape_mismatch,
        'allowed_new_prefixes': list(allowed_new_prefixes),
    }


def configure_sfad_train_scope(model, scope):
    if scope not in ('adapter_only', 'all'):
        raise ValueError(f'Unsupported SFAD train scope: {scope}')
    trainable_names = []
    frozen_names = []
    for name, param in model.named_parameters():
        trainable = scope == 'all' or name.startswith('SFAD_GST')
        param.requires_grad = trainable
        if trainable:
            trainable_names.append(name)
        else:
            frozen_names.append(name)
    return {
        'scope': scope,
        'trainable_param_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'frozen_param_count': sum(p.numel() for p in model.parameters() if not p.requires_grad),
        'trainable_prefixes': sorted({name.split('.')[0] for name in trainable_names}),
        'frozen_prefix_count': len(sorted({name.split('.')[0] for name in frozen_names})),
    }
