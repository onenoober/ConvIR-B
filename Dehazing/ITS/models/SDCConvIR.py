import torch
import torch.nn as nn
import torch.nn.functional as F

from .ConvIR import ConvIR


class SharedDegradationControllerLite(nn.Module):
    def __init__(self, channels):
        super().__init__()
        image_channels = max(16, channels // 8)
        hidden_channels = max(32, channels // 2)
        self.image_proj = nn.Sequential(
            nn.Conv2d(3, image_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.field = nn.Sequential(
            nn.Conv2d(channels * 2 + image_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.affine = nn.Conv2d(1, channels * 2, kernel_size=1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self._last_stats = {}

    def forward(self, image, fused, condition):
        if image.shape[-2:] != fused.shape[-2:]:
            image = F.interpolate(image, size=fused.shape[-2:], mode='bilinear', align_corners=False)
        if condition.shape[-2:] != fused.shape[-2:]:
            condition = F.interpolate(condition, size=fused.shape[-2:], mode='bilinear', align_corners=False)
        image_feature = self.image_proj(image)
        field = self.field(torch.cat([image_feature, fused, condition], dim=1))
        gamma, beta = self.affine(field).chunk(2, dim=1)
        out = fused * (1 + self.alpha * torch.tanh(gamma)) + self.alpha * torch.tanh(beta)
        self._last_stats = self._summarize(field, gamma, beta)
        return out

    def _summarize(self, field, gamma, beta):
        field = field.detach()
        gamma = gamma.detach()
        beta = beta.detach()
        hist = torch.histc(field.float().clamp(0, 1).reshape(-1).cpu(), bins=16, min=0.0, max=1.0)
        prob = hist / hist.sum().clamp_min(1e-12)
        prob = prob[prob > 0]
        entropy = float((-(prob * torch.log(prob)).sum() / torch.log(torch.tensor(16.0))).item())
        return {
            'R_mean': float(field.mean().cpu()),
            'R_std': float(field.std(unbiased=False).cpu()),
            'R_min': float(field.min().cpu()),
            'R_max': float(field.max().cpu()),
            'R_lt_005': float((field < 0.05).float().mean().cpu()),
            'R_gt_095': float((field > 0.95).float().mean().cpu()),
            'R_entropy': entropy,
            'gamma_mean': float(gamma.mean().cpu()),
            'gamma_std': float(gamma.std(unbiased=False).cpu()),
            'beta_mean': float(beta.mean().cpu()),
            'beta_std': float(beta.std(unbiased=False).cpu()),
            'alpha': float(self.alpha.detach().cpu().reshape(-1)[0]),
        }

    def collect_stats(self):
        return dict(self._last_stats)


class SDCLiteConvIR(ConvIR):
    def __init__(self, version, data):
        super().__init__(version, data)
        base_channel = 32
        self.SFAD_SDC = SharedDegradationControllerLite(base_channel * 2)

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
        z = self.SFAD_SDC(x_2, z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

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
        return {'SDC_1_2': self.SFAD_SDC.collect_stats()}


def build_sdc_lite_net(version, data, fam_mode='original'):
    if fam_mode != 'original':
        raise ValueError("SDC-Lite route starts from fam_mode='original'.")
    return SDCLiteConvIR(version, data)


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


def configure_sdc_train_scope(model, scope):
    if scope not in ('adapter_only', 'all'):
        raise ValueError(f'Unsupported SDC train scope: {scope}')
    trainable_names = []
    frozen_names = []
    for name, param in model.named_parameters():
        trainable = scope == 'all' or name.startswith('SFAD_SDC')
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
