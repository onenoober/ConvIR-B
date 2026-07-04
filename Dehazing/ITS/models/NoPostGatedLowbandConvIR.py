import torch
import torch.nn as nn
import torch.nn.functional as F

from .ConvIR import ConvIR


class HaarDWT2D(nn.Module):
    def forward(self, x):
        h, w = x.shape[-2:]
        ph = h % 2
        pw = w % 2
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        a = x[:, :, 0::2, 0::2]
        b = x[:, :, 0::2, 1::2]
        c = x[:, :, 1::2, 0::2]
        d = x[:, :, 1::2, 1::2]
        ll = (a + b + c + d) / 2.0
        lh = (a - b + c - d) / 2.0
        hl = (a + b - c - d) / 2.0
        hh = (a - b - c + d) / 2.0
        return ll, lh, hl, hh, h, w


class HaarIWT2D(nn.Module):
    def forward(self, ll, lh, hl, hh, h, w):
        a = (ll + lh + hl + hh) / 2.0
        b = (ll - lh + hl - hh) / 2.0
        c = (ll + lh - hl - hh) / 2.0
        d = (ll - lh - hl + hh) / 2.0
        out = torch.empty(
            (ll.shape[0], ll.shape[1], ll.shape[2] * 2, ll.shape[3] * 2),
            dtype=ll.dtype,
            device=ll.device,
        )
        out[:, :, 0::2, 0::2] = a
        out[:, :, 0::2, 1::2] = b
        out[:, :, 1::2, 0::2] = c
        out[:, :, 1::2, 1::2] = d
        return out[:, :, :h, :w]


class MidGridActionGate(nn.Module):
    def __init__(self, channels, grid=8, hidden_channels=32, risk_bias=-1.5):
        super().__init__()
        self.grid = grid
        self.context = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )
        self.action = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)
        self.risk = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.action.weight)
        nn.init.zeros_(self.action.bias)
        nn.init.zeros_(self.risk[-1].weight)
        nn.init.constant_(self.risk[-1].bias, risk_bias)

    def forward(self, ll):
        pooled = F.adaptive_avg_pool2d(ll, (self.grid, self.grid))
        hidden = self.context(pooled)
        raw_delta_grid = self.action(hidden)
        unsafe_logit = self.risk(hidden).flatten(1)
        return raw_delta_grid, unsafe_logit


class FinalGridActionGate(nn.Module):
    def __init__(
        self,
        final_channels,
        mid_channels,
        grid=16,
        hidden_channels=32,
        risk_bias=-1.5,
    ):
        super().__init__()
        self.grid = grid
        self.mid_project = nn.Conv2d(mid_channels, hidden_channels, kernel_size=1, bias=True)
        self.global_project = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(final_channels + mid_channels, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(final_channels + hidden_channels * 2, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )
        self.action = nn.Conv2d(hidden_channels, final_channels, kernel_size=1, bias=True)
        self.risk = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.action.weight)
        nn.init.zeros_(self.action.bias)
        nn.init.zeros_(self.risk[-1].weight)
        nn.init.constant_(self.risk[-1].bias, risk_bias)

    def forward(self, final_ll, mid_feature):
        pooled_final = F.adaptive_avg_pool2d(final_ll, (self.grid, self.grid))
        pooled_mid = F.adaptive_avg_pool2d(mid_feature, (self.grid, self.grid))
        mid_ctx = self.mid_project(pooled_mid)
        final_global = F.adaptive_avg_pool2d(final_ll, 1)
        mid_global = F.adaptive_avg_pool2d(mid_feature, 1)
        global_ctx = self.global_project(torch.cat([final_global, mid_global], dim=1))
        global_ctx = global_ctx.expand(-1, -1, self.grid, self.grid)
        hidden = self.context(torch.cat([pooled_final, mid_ctx, global_ctx], dim=1))
        raw_delta_grid = self.action(hidden)
        unsafe_logit = self.risk(hidden).flatten(1)
        return raw_delta_grid, unsafe_logit


class GatedMidFinalLowbandPolicy(nn.Module):
    def __init__(
        self,
        mid_channels=64,
        final_channels=32,
        hidden_channels=32,
        mid_grid=8,
        final_grid=16,
        risk_gamma=0.5,
        risk_bias=-1.5,
    ):
        super().__init__()
        self.risk_gamma = risk_gamma
        self.mid_policy = MidGridActionGate(mid_channels, mid_grid, hidden_channels, risk_bias)
        self.final_policy = FinalGridActionGate(final_channels, mid_channels, final_grid, hidden_channels, risk_bias)
        self.dwt = HaarDWT2D()
        self.iwt = HaarIWT2D()
        self.last_tensors = {}

    def _scale_from_logit(self, unsafe_logit):
        unsafe_prob = torch.sigmoid(unsafe_logit)
        scale = (1.0 - unsafe_prob).clamp(0.0, 1.0) ** self.risk_gamma
        return unsafe_prob, scale

    def _apply_lowband(self, z, raw_delta_grid, scale):
        ll, lh, hl, hh, h, w = self.dwt(z)
        delta_grid = raw_delta_grid * scale.view(-1, 1, 1, 1)
        delta_ll = F.interpolate(delta_grid, size=ll.shape[-2:], mode="bilinear", align_corners=False)
        zero_lh = torch.zeros_like(lh)
        zero_hl = torch.zeros_like(hl)
        zero_hh = torch.zeros_like(hh)
        delta = self.iwt(delta_ll, zero_lh, zero_hl, zero_hh, h, w)
        return z + delta, delta_grid, delta

    def forward_mid(self, mid):
        ll, _, _, _, _, _ = self.dwt(mid)
        raw_delta_grid, unsafe_logit = self.mid_policy(ll)
        unsafe_prob, scale = self._scale_from_logit(unsafe_logit)
        out, scaled_delta_grid, delta = self._apply_lowband(mid, raw_delta_grid, scale)
        self.last_tensors.update(
            {
                "mid_raw_delta_grid": raw_delta_grid,
                "mid_scaled_delta_grid": scaled_delta_grid,
                "mid_unsafe_logit": unsafe_logit,
                "mid_unsafe_prob": unsafe_prob,
                "mid_scale": scale,
                "mid_delta": delta,
            }
        )
        return out

    def forward_final(self, final, mid_context):
        ll, _, _, _, _, _ = self.dwt(final)
        raw_delta_grid, unsafe_logit = self.final_policy(ll, mid_context)
        unsafe_prob, scale = self._scale_from_logit(unsafe_logit)
        out, scaled_delta_grid, delta = self._apply_lowband(final, raw_delta_grid, scale)
        self.last_tensors.update(
            {
                "final_raw_delta_grid": raw_delta_grid,
                "final_scaled_delta_grid": scaled_delta_grid,
                "final_unsafe_logit": unsafe_logit,
                "final_unsafe_prob": unsafe_prob,
                "final_scale": scale,
                "final_delta": delta,
            }
        )
        return out

    def tensor_stats(self):
        stats = {}
        for key, value in self.last_tensors.items():
            if not torch.is_tensor(value):
                continue
            detached = value.detach()
            stats[f"{key}_mean"] = float(detached.mean().cpu())
            stats[f"{key}_std"] = float(detached.std(unbiased=False).cpu())
            stats[f"{key}_abs_mean"] = float(detached.abs().mean().cpu())
            stats[f"{key}_rms"] = float(torch.sqrt(torch.mean(detached.float() ** 2)).cpu())
        return stats


class NoPostGatedLowbandConvIR(ConvIR):
    def __init__(
        self,
        version,
        data,
        hidden_channels=32,
        mid_grid=8,
        final_grid=16,
        risk_gamma=0.5,
        risk_bias=-1.5,
    ):
        super().__init__(version, data)
        self.nopost_gated_lowband_policy = GatedMidFinalLowbandPolicy(
            mid_channels=64,
            final_channels=32,
            hidden_channels=hidden_channels,
            mid_grid=mid_grid,
            final_grid=final_grid,
            risk_gamma=risk_gamma,
            risk_bias=risk_bias,
        )

    def forward(self, x):
        self.nopost_gated_lowband_policy.last_tensors = {}
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

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z = self.nopost_gated_lowband_policy.forward_mid(z)
        mid_context = z
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_ + x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.nopost_gated_lowband_policy.forward_final(z, mid_context)
        z = self.feat_extract[5](z)
        outputs.append(z + x)

        return outputs

    def collect_modulation_stats(self, x):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            _ = self(x)
            stats = self.nopost_gated_lowband_policy.tensor_stats()
        if was_training:
            self.train()
        return {"nopost_gated_lowband_policy": stats}


def build_net(
    version,
    data,
    fam_mode="original",
    hidden_channels=32,
    mid_grid=8,
    final_grid=16,
    risk_gamma=0.5,
    risk_bias=-1.5,
):
    if fam_mode != "original":
        raise ValueError("NoPost gated lowband policy starts from official ConvIR-B original FAM.")
    return NoPostGatedLowbandConvIR(
        version,
        data,
        hidden_channels=hidden_channels,
        mid_grid=mid_grid,
        final_grid=final_grid,
        risk_gamma=risk_gamma,
        risk_bias=risk_bias,
    )


def load_haze4k_partial(model, checkpoint_state, allowed_new_prefixes=("nopost_gated_lowband_policy.",)):
    model_state = model.state_dict()
    loaded = {}
    unexpected = []
    shape_mismatch = []
    for key, value in checkpoint_state.items():
        if key not in model_state:
            unexpected.append(key)
        elif tuple(model_state[key].shape) != tuple(value.shape):
            shape_mismatch.append(
                {
                    "key": key,
                    "checkpoint_shape": list(value.shape),
                    "model_shape": list(model_state[key].shape),
                }
            )
        else:
            loaded[key] = value

    missing = [key for key in model_state if key not in loaded]
    bad_missing = [
        key
        for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_new_prefixes)
    ]
    if unexpected or shape_mismatch or bad_missing:
        raise RuntimeError(
            "partial-load failed: "
            f"unexpected={unexpected[:20]} "
            f"shape_mismatch={shape_mismatch[:20]} "
            f"bad_missing={bad_missing[:20]}"
        )
    model_state.update(loaded)
    model.load_state_dict(model_state, strict=True)
    return {
        "loaded_count": len(loaded),
        "missing_new_module_count": len(missing),
        "missing_new_modules": sorted(missing),
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
        "allowed_new_prefixes": list(allowed_new_prefixes),
    }
