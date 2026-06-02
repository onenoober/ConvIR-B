import torch
import torch.nn.functional as F

from .ConvIR import ConvIR
from .haze_priors import HazePriorResidual
from .pfd_modules import LowPassResidualDelta, ResidualHardFeatureDelta


class PFDConvIR(ConvIR):
    def __init__(
        self,
        version,
        data,
        pfd_rhfd=False,
        pfd_hscm=False,
        pfd_pffb=False,
        pfd_pffb_high=False,
        pfd_teacher=False,
    ):
        if pfd_pffb_high:
            raise ValueError("PFFB-High is deferred for the PFD mainline.")
        if pfd_teacher:
            raise ValueError("Teacher preservation is conditional and not implemented as a default flag.")
        super(PFDConvIR, self).__init__(version, data, fam_mode="original")
        self.pfd_rhfd = bool(pfd_rhfd)
        self.pfd_hscm = bool(pfd_hscm)
        self.pfd_pffb = bool(pfd_pffb)
        self.pfd_pffb_high = bool(pfd_pffb_high)
        self.pfd_teacher = bool(pfd_teacher)

        self.PFD_HSCM2 = HazePriorResidual(64)
        self.PFD_HSCM1 = HazePriorResidual(128)
        self.PFD_RHFD2 = ResidualHardFeatureDelta(64)
        self.PFD_RHFD1 = ResidualHardFeatureDelta(128)
        self.PFD_PFFB2 = LowPassResidualDelta(64)
        self.PFD_PFFB1 = LowPassResidualDelta(128)

    def _apply_stage(self, x, rhfd, pffb):
        if self.pfd_rhfd:
            x = x + rhfd(x)
        if self.pfd_pffb:
            x = x + pffb(x)
        return x

    def _stage_delta_stats(self, feature, module):
        delta = module(feature)
        feature_norm = feature.norm().item()
        delta_norm = delta.norm().item()
        return {
            "delta_abs_mean": delta.abs().mean().item(),
            "delta_abs_max": delta.abs().max().item(),
            "delta_std": delta.std(unbiased=False).item(),
            "delta_norm": delta_norm,
            "feature_norm": feature_norm,
            "delta_norm_ratio": delta_norm / max(feature_norm, 1e-12),
        }

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)
        if self.pfd_hscm:
            z2 = z2 + self.PFD_HSCM2(x_2)
            z4 = z4 + self.PFD_HSCM1(x_4)

        outputs = list()
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        z = self._apply_stage(z, self.PFD_RHFD2, self.PFD_PFFB2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self._apply_stage(z, self.PFD_RHFD1, self.PFD_PFFB1)
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

    def collect_pfd_stats(self, x):
        with torch.no_grad():
            x_2 = F.interpolate(x, scale_factor=0.5)
            x_4 = F.interpolate(x_2, scale_factor=0.5)
            z2 = self.SCM2(x_2)
            z4 = self.SCM1(x_4)
            stats = {
                "flags": {
                    "rhfd": self.pfd_rhfd,
                    "hscm": self.pfd_hscm,
                    "pffb": self.pfd_pffb,
                    "pffb_high": self.pfd_pffb_high,
                    "teacher": self.pfd_teacher,
                }
            }
            if self.pfd_hscm:
                stats["HSCM2"] = self.PFD_HSCM2.stats(x_2)
                stats["HSCM1"] = self.PFD_HSCM1.stats(x_4)
                z2 = z2 + self.PFD_HSCM2(x_2)
                z4 = z4 + self.PFD_HSCM1(x_4)

            x_ = self.feat_extract[0](x)
            res1 = self.Encoder[0](x_)
            z = self.feat_extract[1](res1)
            z = self.FAM2(z, z2)
            if self.pfd_rhfd:
                stats["RHFD2"] = self._stage_delta_stats(z, self.PFD_RHFD2)
                z = z + self.PFD_RHFD2(z)
            if self.pfd_pffb:
                stats["PFFB2"] = self._stage_delta_stats(z, self.PFD_PFFB2)
                z = z + self.PFD_PFFB2(z)
            res2 = self.Encoder[1](z)

            z = self.feat_extract[2](res2)
            z = self.FAM1(z, z4)
            if self.pfd_rhfd:
                stats["RHFD1"] = self._stage_delta_stats(z, self.PFD_RHFD1)
                z = z + self.PFD_RHFD1(z)
            if self.pfd_pffb:
                stats["PFFB1"] = self._stage_delta_stats(z, self.PFD_PFFB1)
                z = z + self.PFD_PFFB1(z)
            return stats


def build_pfd_net(
    version,
    data,
    pfd_rhfd=False,
    pfd_hscm=False,
    pfd_pffb=False,
    pfd_pffb_high=False,
    pfd_teacher=False,
):
    return PFDConvIR(
        version,
        data,
        pfd_rhfd=pfd_rhfd,
        pfd_hscm=pfd_hscm,
        pfd_pffb=pfd_pffb,
        pfd_pffb_high=pfd_pffb_high,
        pfd_teacher=pfd_teacher,
    )
