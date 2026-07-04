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


class ActionConditionedLowFrequencyBank(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels=32,
        grid=8,
        context_channels=0,
        delta_scale=0.25,
        coverage_budget=0.35,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.grid = grid
        self.delta_scale = delta_scale
        self.coverage_budget = coverage_budget
        self.dwt = HaarDWT2D()
        self.iwt = HaarIWT2D()

        state_channels = channels + context_channels
        self.state_encoder = nn.Sequential(
            nn.Conv2d(state_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )
        self.mild_head = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)
        self.medium_head = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)
        self.strong_head = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)
        self.state_project = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
        )
        self.action_selector = nn.Sequential(
            nn.Linear(hidden_channels + 8, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor([0.0, -4.0, -5.0, -6.0], dtype=torch.float32).view(1, 4),
        )
        self.last_stats = {}
        self._init_identity()

    def _init_identity(self):
        for head in (self.mild_head, self.medium_head, self.strong_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.action_selector[-1].weight)
        nn.init.zeros_(self.action_selector[-1].bias)

    def _bounded_heads(self, hidden, ll):
        flat = ll.detach().flatten(2)
        scale = flat.std(dim=2).view(ll.shape[0], ll.shape[1], 1, 1).clamp_min(1e-4)
        scale = scale * self.delta_scale
        return [
            torch.zeros(
                (ll.shape[0], ll.shape[1], self.grid, self.grid),
                dtype=ll.dtype,
                device=ll.device,
            ),
            torch.tanh(self.mild_head(hidden)) * scale,
            torch.tanh(self.medium_head(hidden)) * scale,
            torch.tanh(self.strong_head(hidden)) * scale,
        ]

    def _action_features(self, deltas, ll, state_vec):
        rows = []
        ll_grid = F.adaptive_avg_pool2d(ll.detach(), (self.grid, self.grid))
        ll_energy = ll_grid.flatten(1).norm(dim=1).clamp_min(1e-8)
        strengths = deltas[0].new_tensor([0.0, 1.0, 2.0, 3.0])
        for idx, delta in enumerate(deltas):
            delta_flat = delta.flatten(1)
            delta_abs = delta.abs()
            delta_norm = delta_flat.norm(dim=1)
            align = (delta_flat * ll_grid.flatten(1)).sum(dim=1) / (delta_norm * ll_energy + 1e-8)
            stats = torch.stack(
                [
                    delta_abs.mean(dim=(1, 2, 3)),
                    torch.sqrt(torch.mean(delta.float() ** 2, dim=(1, 2, 3))).to(delta.dtype),
                    delta_abs.amax(dim=(1, 2, 3)),
                    align,
                    torch.full_like(align, strengths[idx]),
                    torch.full_like(align, float(idx == 0)),
                    torch.full_like(align, float(idx >= 2)),
                    torch.full_like(align, float(idx == 3)),
                ],
                dim=1,
            )
            rows.append(torch.cat([state_vec, stats], dim=1))
        return rows

    def forward(self, z, context=None):
        ll, lh, hl, hh, h, w = self.dwt(z)
        pooled_ll = F.adaptive_avg_pool2d(ll, (self.grid, self.grid))
        if context is not None:
            ctx = F.adaptive_avg_pool2d(context, (self.grid, self.grid))
            state = torch.cat([pooled_ll, ctx], dim=1)
        else:
            state = pooled_ll
        hidden = self.state_encoder(state)
        deltas = self._bounded_heads(hidden, ll)
        state_vec = self.state_project(hidden).flatten(1)
        selector_rows = self._action_features(deltas, ll, state_vec)
        logits = torch.cat([self.action_selector(row) for row in selector_rows], dim=1)
        logits = logits + self.action_bias.to(dtype=logits.dtype, device=logits.device)
        gate = torch.softmax(logits, dim=1)

        mixture = torch.zeros_like(deltas[0])
        for idx in range(1, 4):
            mixture = mixture + gate[:, idx].view(-1, 1, 1, 1) * deltas[idx]
        mixture = mixture * self.coverage_budget
        mixture_ll = F.interpolate(mixture, size=ll.shape[-2:], mode="bilinear", align_corners=False)
        out = self.iwt(ll + mixture_ll, lh, hl, hh, h, w)
        with torch.no_grad():
            action_mass = gate[:, 1:].sum(dim=1)
            self.last_stats = {
                "noop_gate_mean": float(gate[:, 0].mean().detach().cpu()),
                "mild_gate_mean": float(gate[:, 1].mean().detach().cpu()),
                "medium_gate_mean": float(gate[:, 2].mean().detach().cpu()),
                "strong_gate_mean": float(gate[:, 3].mean().detach().cpu()),
                "action_mass_mean": float(action_mass.mean().detach().cpu()),
                "mixture_abs_mean": float(mixture.abs().mean().detach().cpu()),
                "mixture_rms": float(torch.sqrt(torch.mean(mixture.float() ** 2)).detach().cpu()),
            }
        return out


class IntegratedLowFrequencyRestoration(nn.Module):
    def __init__(self, hidden_channels=32, delta_scale=0.25, coverage_budget=0.35):
        super().__init__()
        self.bottleneck = ActionConditionedLowFrequencyBank(
            128,
            hidden_channels=hidden_channels,
            grid=4,
            context_channels=0,
            delta_scale=delta_scale,
            coverage_budget=coverage_budget,
        )
        self.early = ActionConditionedLowFrequencyBank(
            128,
            hidden_channels=hidden_channels,
            grid=4,
            context_channels=128,
            delta_scale=delta_scale,
            coverage_budget=coverage_budget,
        )
        self.mid = ActionConditionedLowFrequencyBank(
            64,
            hidden_channels=hidden_channels,
            grid=8,
            context_channels=128,
            delta_scale=delta_scale,
            coverage_budget=coverage_budget,
        )
        self.final = ActionConditionedLowFrequencyBank(
            32,
            hidden_channels=hidden_channels,
            grid=16,
            context_channels=64,
            delta_scale=delta_scale,
            coverage_budget=coverage_budget,
        )

    def reset_stats(self):
        for block in (self.bottleneck, self.early, self.mid, self.final):
            block.last_stats = {}

    def tensor_stats(self):
        stats = {}
        for name, block in (
            ("bottleneck", self.bottleneck),
            ("early", self.early),
            ("mid", self.mid),
            ("final", self.final),
        ):
            for key, value in block.last_stats.items():
                stats[f"{name}_{key}"] = value
        return stats


class ILFRBACSConvIR(ConvIR):
    def __init__(self, version, data, hidden_channels=32, delta_scale=0.25, coverage_budget=0.35):
        super().__init__(version, data)
        self.ilfrb_acs = IntegratedLowFrequencyRestoration(
            hidden_channels=hidden_channels,
            delta_scale=delta_scale,
            coverage_budget=coverage_budget,
        )

    def forward(self, x):
        self.ilfrb_acs.reset_stats()
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

        z = self.ilfrb_acs.bottleneck(z)
        z = self.Decoder[0](z)
        z = self.ilfrb_acs.early(z, context=z)
        early_context = z
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_ + x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z = self.ilfrb_acs.mid(z, context=early_context)
        mid_context = z
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_ + x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.ilfrb_acs.final(z, context=mid_context)
        z = self.feat_extract[5](z)
        outputs.append(z + x)
        return outputs

    def collect_modulation_stats(self, x):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            _ = self(x)
            stats = self.ilfrb_acs.tensor_stats()
        if was_training:
            self.train()
        return {"ilfrb_acs": stats}


def build_net(
    version,
    data,
    fam_mode="original",
    hidden_channels=32,
    delta_scale=0.25,
    coverage_budget=0.35,
):
    if fam_mode != "original":
        raise ValueError("ILFRB-ACS starts from official ConvIR-B original FAM.")
    return ILFRBACSConvIR(
        version,
        data,
        hidden_channels=hidden_channels,
        delta_scale=delta_scale,
        coverage_budget=coverage_budget,
    )


def load_haze4k_partial(model, checkpoint_state, allowed_new_prefixes=("ilfrb_acs.",)):
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
