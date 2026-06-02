import torch
import torch.nn.functional as F

from .ConvIR import ConvIR
from .apdr_modules import APDRScaleAdapter


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
    ):
        if apdr_prior_mode != "rgb_haze":
            raise ValueError("APDR-v0 only supports apdr_prior_mode='rgb_haze'.")
        super(APDRConvIR, self).__init__(version, data, fam_mode="original")
        self.apdr_prior_mode = apdr_prior_mode
        self.apdr_force_zero_gate = bool(apdr_force_zero_gate)

        self.APDR_4 = APDRScaleAdapter(
            128,
            residual_max=apdr_residual_max,
            gate_max=apdr_gate_max,
            gate_init=apdr_gate_init,
        )
        self.APDR_2 = APDRScaleAdapter(
            64,
            residual_max=apdr_residual_max,
            gate_max=apdr_gate_max,
            gate_init=apdr_gate_init,
        )
        self.APDR_1 = APDRScaleAdapter(
            32,
            residual_max=apdr_residual_max,
            gate_max=apdr_gate_max,
            gate_init=apdr_gate_init,
        )
        self._last_apdr_tensors = None

    def _adapt(self, adapter, hazy, anchor, feature):
        output, tensors = adapter(
            hazy,
            anchor,
            feature,
            force_zero_gate=self.apdr_force_zero_gate,
        )
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
        out_4, stats_4 = self._adapt(self.APDR_4, x_4, z_ + x_4, z)
        outputs.append(out_4)
        apdr_tensors.append(stats_4)

        z = self.feat_extract[3](z)
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        out_2, stats_2 = self._adapt(self.APDR_2, x_2, z_ + x_2, z)
        outputs.append(out_2)
        apdr_tensors.append(stats_2)

        z = self.feat_extract[4](z)
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        full_feature = z
        z = self.feat_extract[5](z)
        out_1, stats_1 = self._adapt(self.APDR_1, x, z + x, full_feature)
        outputs.append(out_1)
        apdr_tensors.append(stats_1)

        self._last_apdr_tensors = apdr_tensors
        return outputs

    def apdr_regularization(self):
        if not self._last_apdr_tensors:
            return {}
        gates = [item["gate"] for item in self._last_apdr_tensors]
        residuals = [item["residual"] for item in self._last_apdr_tensors]
        return {
            "apdr_anchor": sum(residual.abs().mean() for residual in residuals) / len(residuals),
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
                    "gate_mean": gate.mean().item(),
                    "gate_std": gate.std(unbiased=False).item(),
                    "gate_min": gate.min().item(),
                    "gate_max": gate.max().item(),
                    "residual_abs_mean": residual.abs().mean().item(),
                    "residual_abs_max": residual.abs().max().item(),
                    "residual_raw_abs_mean": residual_raw.abs().mean().item(),
                    "anchor_abs_mean": anchor.abs().mean().item(),
                }
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
):
    return APDRConvIR(
        version,
        data,
        apdr_prior_mode=apdr_prior_mode,
        apdr_residual_max=apdr_residual_max,
        apdr_gate_max=apdr_gate_max,
        apdr_gate_init=apdr_gate_init,
        apdr_force_zero_gate=apdr_force_zero_gate,
    )
