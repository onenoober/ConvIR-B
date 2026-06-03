import torch
import torch.nn.functional as F

from .ConvIR import ConvIR
from .apdr_modules import APDRScaleAdapter, APDRV02RScaleAdapter, APDRV02ScaleAdapter


def _active_scale_set(active_scales):
    if active_scales == "all":
        return {"quarter", "half", "full"}
    if active_scales == "full":
        return {"full"}
    raise ValueError(f"Unsupported apdr_active_scales: {active_scales}")


class APDRConvIR(ConvIR):
    def __init__(
        self,
        version,
        data,
        apdr_prior_mode="rgb_haze",
        apdr_residual_max=0.04,
        apdr_gate_max=0.5,
        apdr_gate_init=0.02,
        apdr_force_zero_gate=False,
        apdr_active_scales="all",
        apdr_selector_mode="v0",
        apdr_residual_capacity="linear",
    ):
        if apdr_prior_mode != "rgb_haze":
            raise ValueError("APDR only supports apdr_prior_mode='rgb_haze'.")
        if apdr_selector_mode not in ("v0", "v0_2", "v0_2r"):
            raise ValueError(f"Unsupported apdr_selector_mode: {apdr_selector_mode}")
        if apdr_residual_capacity not in ("linear", "shallow_mlp"):
            raise ValueError(f"Unsupported apdr_residual_capacity: {apdr_residual_capacity}")
        super(APDRConvIR, self).__init__(version, data, fam_mode="original")
        self.apdr_prior_mode = apdr_prior_mode
        self.apdr_force_zero_gate = bool(apdr_force_zero_gate)
        self.apdr_active_scales = apdr_active_scales
        self.apdr_selector_mode = apdr_selector_mode
        self.apdr_residual_capacity = apdr_residual_capacity
        self._active_scale_names = _active_scale_set(apdr_active_scales)
        if apdr_selector_mode == "v0_2":
            adapter_cls = APDRV02ScaleAdapter
        elif apdr_selector_mode == "v0_2r":
            adapter_cls = APDRV02RScaleAdapter
        else:
            adapter_cls = APDRScaleAdapter

        self.APDR_4 = adapter_cls(
            128,
            residual_max=apdr_residual_max,
            gate_max=apdr_gate_max,
            gate_init=apdr_gate_init,
            residual_capacity=apdr_residual_capacity,
        )
        self.APDR_2 = adapter_cls(
            64,
            residual_max=apdr_residual_max,
            gate_max=apdr_gate_max,
            gate_init=apdr_gate_init,
            residual_capacity=apdr_residual_capacity,
        )
        self.APDR_1 = adapter_cls(
            32,
            residual_max=apdr_residual_max,
            gate_max=apdr_gate_max,
            gate_init=apdr_gate_init,
            residual_capacity=apdr_residual_capacity,
        )
        self._last_apdr_tensors = None

    def _adapt(self, name, adapter, hazy, anchor, feature):
        active = name in self._active_scale_names
        output, tensors = adapter(
            hazy,
            anchor,
            feature,
            force_zero_gate=self.apdr_force_zero_gate or not active,
        )
        tensors["scale"] = name
        tensors["active"] = active
        return output, tensors

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        apdr_tensors = []
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
        out_4, stats_4 = self._adapt("quarter", self.APDR_4, x_4, z_ + x_4, z)
        outputs.append(out_4)
        apdr_tensors.append(stats_4)

        z = self.feat_extract[3](z)
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        out_2, stats_2 = self._adapt("half", self.APDR_2, x_2, z_ + x_2, z)
        outputs.append(out_2)
        apdr_tensors.append(stats_2)

        z = self.feat_extract[4](z)
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        full_feature = z
        z = self.feat_extract[5](z)
        out_1, stats_1 = self._adapt("full", self.APDR_1, x, z + x, full_feature)
        outputs.append(out_1)
        apdr_tensors.append(stats_1)

        self._last_apdr_tensors = apdr_tensors
        return outputs

    def active_apdr_prefixes(self):
        prefixes = []
        if "quarter" in self._active_scale_names:
            prefixes.append("APDR_4")
        if "half" in self._active_scale_names:
            prefixes.append("APDR_2")
        if "full" in self._active_scale_names:
            prefixes.append("APDR_1")
        return tuple(prefixes)

    def apdr_regularization(self):
        if not self._last_apdr_tensors:
            return {}
        active_items = [item for item in self._last_apdr_tensors if item.get("active", True)]
        if not active_items:
            return {}
        gates = [item["gate"] for item in active_items]
        residuals = [item["residual"] for item in active_items]
        return {
            "apdr_anchor": sum(residual.abs().mean() for residual in residuals) / len(residuals),
            "apdr_gate": sum(gate.mean() for gate in gates) / len(gates),
            "apdr_residual": sum(residual.abs().mean() for residual in residuals) / len(residuals),
        }

    def apdr_training_regularization(self, targets, risk_temperature=5.0, eps=1e-6):
        if not self._last_apdr_tensors:
            return {}
        active_items = [item for item in self._last_apdr_tensors if item.get("active", True)]
        if not active_items:
            return {}

        target_by_scale = {
            "quarter": targets[0],
            "half": targets[1],
            "full": targets[2],
        }
        anchor_terms = []
        delta_terms = []
        gate_supervision_terms = []
        gates = []
        residuals = []

        for item in active_items:
            target = target_by_scale[item["scale"]]
            anchor = item["anchor"].detach()
            output = item["output"]
            gate = item["gate"]
            residual = item["residual"]
            residual_raw = item["residual_raw"]

            error = (anchor - target).abs().mean(dim=1, keepdim=True).detach()
            denom = error.amax(dim=(2, 3), keepdim=True).clamp_min(eps)
            risk = (error / denom).clamp(0.0, 1.0)
            safe_weight = torch.exp(-float(risk_temperature) * risk)
            delta_star = (target - anchor).clamp(
                min=-float(item["residual_max"]),
                max=float(item["residual_max"]),
            ).detach()
            delta_weight = (gate / float(item["gate_max"])).detach().clamp(0.0, 1.0)

            anchor_terms.append((safe_weight * (output - anchor).abs()).mean())
            delta_terms.append((delta_weight * (residual_raw - delta_star).abs()).mean())
            gate_supervision_terms.append(F.l1_loss(gate / item["gate_max"], risk))
            gates.append(gate)
            residuals.append(residual)

        return {
            "apdr_anchor": sum(anchor_terms) / len(anchor_terms),
            "apdr_delta_supervision": sum(delta_terms) / len(delta_terms),
            "apdr_gate_supervision": sum(gate_supervision_terms) / len(gate_supervision_terms),
            "apdr_gate": sum(gate.mean() for gate in gates) / len(gates),
            "apdr_residual": sum(residual.abs().mean() for residual in residuals) / len(residuals),
        }

    def collect_apdr_stats(self, x):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            self.forward(x)
            items = self._last_apdr_tensors or []
            stats = {"scales": {}}
            for name, item in zip(("quarter", "half", "full"), items):
                gate = item["gate"]
                residual = item["residual"]
                residual_raw = item["residual_raw"]
                anchor = item["anchor"]
                stats["scales"][name] = {
                    "active": 1.0 if item.get("active", True) else 0.0,
                    "gate_mean": gate.mean().item(),
                    "gate_std": gate.std(unbiased=False).item(),
                    "gate_min": gate.min().item(),
                    "gate_max": gate.max().item(),
                    "residual_abs_mean": residual.abs().mean().item(),
                    "residual_abs_max": residual.abs().max().item(),
                    "residual_raw_abs_mean": residual_raw.abs().mean().item(),
                    "anchor_abs_mean": anchor.abs().mean().item(),
                }
                if "global_gate" in item:
                    global_gate = item["global_gate"]
                    spatial_gate = item["spatial_gate"]
                    stats["scales"][name].update(
                        {
                            "global_gate_mean": global_gate.mean().item(),
                            "global_gate_min": global_gate.min().item(),
                            "global_gate_max": global_gate.max().item(),
                            "spatial_gate_mean": spatial_gate.mean().item(),
                            "spatial_gate_std": spatial_gate.std(unbiased=False).item(),
                        }
                    )
        self.train(was_training)
        return stats


def build_apdr_net(
    version,
    data,
    apdr_prior_mode="rgb_haze",
    apdr_residual_max=0.04,
    apdr_gate_max=0.5,
    apdr_gate_init=0.02,
    apdr_force_zero_gate=False,
    apdr_active_scales="all",
    apdr_selector_mode="v0",
    apdr_residual_capacity="linear",
):
    return APDRConvIR(
        version,
        data,
        apdr_prior_mode=apdr_prior_mode,
        apdr_residual_max=apdr_residual_max,
        apdr_gate_max=apdr_gate_max,
        apdr_gate_init=apdr_gate_init,
        apdr_force_zero_gate=apdr_force_zero_gate,
        apdr_active_scales=apdr_active_scales,
        apdr_selector_mode=apdr_selector_mode,
        apdr_residual_capacity=apdr_residual_capacity,
    )
