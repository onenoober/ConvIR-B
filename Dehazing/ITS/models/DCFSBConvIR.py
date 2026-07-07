import torch
import torch.nn as nn
import torch.nn.functional as F

from .ConvIR import ConvIR


class DCFSBBottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(32, channels // 4)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, 2, kernel_size=1),
            nn.Sigmoid(),
        )
        self.low_adapter = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.high_adapter = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.alpha_low = nn.Parameter(torch.zeros(1))
        self.alpha_high = nn.Parameter(torch.zeros(1))
        self._last_stats = {}

    def _lowpass(self, x):
        return F.avg_pool2d(x, kernel_size=5, stride=1, padding=2, count_include_pad=False)

    def forward(self, x):
        low = self._lowpass(x)
        high = x - low
        descriptor = torch.cat([low, high, torch.abs(high)], dim=1)
        gates = self.gate(descriptor)
        low_gate = gates[:, 0:1]
        high_gate = gates[:, 1:2]
        low_update = self.low_adapter(low) * low_gate
        high_update = self.high_adapter(high) * high_gate
        out = x + self.alpha_low * low_update + self.alpha_high * high_update
        self._last_stats = self._summarize(low, high, low_gate, high_gate, low_update, high_update)
        return out

    def _summarize(self, low, high, low_gate, high_gate, low_update, high_update):
        low = low.detach()
        high = high.detach()
        low_gate = low_gate.detach()
        high_gate = high_gate.detach()
        low_update = low_update.detach()
        high_update = high_update.detach()
        low_energy = low.abs().mean().clamp_min(1e-12)
        high_energy = high.abs().mean()
        return {
            'low_energy': float(low_energy.cpu()),
            'high_energy': float(high_energy.cpu()),
            'high_low_energy_ratio': float((high_energy / low_energy).cpu()),
            'low_gate_mean': float(low_gate.mean().cpu()),
            'low_gate_std': float(low_gate.std(unbiased=False).cpu()),
            'high_gate_mean': float(high_gate.mean().cpu()),
            'high_gate_std': float(high_gate.std(unbiased=False).cpu()),
            'low_update_abs_mean': float(low_update.abs().mean().cpu()),
            'high_update_abs_mean': float(high_update.abs().mean().cpu()),
            'alpha_low': float(self.alpha_low.detach().cpu().reshape(-1)[0]),
            'alpha_high': float(self.alpha_high.detach().cpu().reshape(-1)[0]),
            'alpha_abs_mean': float((self.alpha_low.detach().abs() + self.alpha_high.detach().abs()).cpu().reshape(-1)[0] / 2.0),
        }

    def collect_stats(self):
        return dict(self._last_stats)


class DCFSBConvIR(ConvIR):
    def __init__(self, version, data):
        super().__init__(version, data)
        base_channel = 32
        self.DCFSB_Bottleneck = DCFSBBottleneck(base_channel * 4)

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
        z = self.DCFSB_Bottleneck(z)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)

        z = self.feat_extract[3](z)
        outputs.append(z_ + x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)

        z = self.feat_extract[4](z)
        outputs.append(z_ + x_2)

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
        return {'DCFSB_Bottleneck': self.DCFSB_Bottleneck.collect_stats()}


def build_dcfsb_bottleneck_net(version, data, fam_mode='original'):
    if fam_mode != 'original':
        raise ValueError("DCFSB route starts from fam_mode='original'.")
    return DCFSBConvIR(version, data)


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
    bad_missing = [key for key in missing if not any(key.startswith(prefix) for prefix in allowed_new_prefixes)]
    if unexpected or shape_mismatch or bad_missing:
        raise RuntimeError(f'partial-load failed: unexpected={unexpected}, shape_mismatch={shape_mismatch}, bad_missing={bad_missing}')
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


def configure_dcfsb_train_scope(model, scope):
    if scope not in ('adapter_only', 'all'):
        raise ValueError(f'Unsupported DCFSB train scope: {scope}')
    trainable_names = []
    frozen_names = []
    for name, param in model.named_parameters():
        trainable = scope == 'all' or name.startswith('DCFSB_Bottleneck')
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
