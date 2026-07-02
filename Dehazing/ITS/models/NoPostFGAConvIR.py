import torch
import torch.nn as nn
import torch.nn.functional as F

from .ConvIR import DBlock, EBlock, FAM, SCM
from .layers import BasicConv
from .nopost_fga import NoPostPBCFGA, summarize_aux


class NoPostFGAConvIR(nn.Module):
    def __init__(
        self,
        version,
        data,
        nopost_use_low=True,
        nopost_use_detail=False,
        nopost_gate_bias=-3.0,
        nopost_context_channels=32,
        nopost_hidden_channels=32,
    ):
        super().__init__()

        if version == "small":
            num_res = 4
        elif version == "base":
            num_res = 8
        elif version == "large":
            num_res = 16
        else:
            raise ValueError(f"Unsupported ConvIR version: {version}")

        base_channel = 32

        self.Encoder = nn.ModuleList(
            [
                EBlock(base_channel, num_res, data),
                EBlock(base_channel * 2, num_res, data),
                EBlock(base_channel * 4, num_res, data),
            ]
        )

        self.feat_extract = nn.ModuleList(
            [
                BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
                BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
                BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
                BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
                BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
                BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.Decoder = nn.ModuleList(
            [
                DBlock(base_channel * 4, num_res, data),
                DBlock(base_channel * 2, num_res, data),
                DBlock(base_channel, num_res, data),
            ]
        )

        self.Convs = nn.ModuleList(
            [
                BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
                BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
            ]
        )

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.nopost_adapter = NoPostPBCFGA(
            feature_channels=base_channel,
            context_channels=nopost_context_channels,
            hidden_channels=nopost_hidden_channels,
            gate_bias=nopost_gate_bias,
            use_low=nopost_use_low,
            use_detail=nopost_use_detail,
        )
        self._last_nopost_aux = None

    def _forward_to_final_feature(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = []
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
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_ + x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        final_feature = self.Decoder[2](z)
        features = {
            "res1": res1,
            "res2": res2,
            "scm2": z2,
            "scm4": z4,
            "final_feature": final_feature,
        }
        return outputs, features

    def extract_nopost_features(self, x):
        _, features = self._forward_to_final_feature(x)
        return features

    def forward(self, x, return_aux=False):
        outputs, features = self._forward_to_final_feature(x)
        final_feature = features["final_feature"]
        delta_feature, aux = self.nopost_adapter(
            hazy=x,
            final_feature=final_feature,
            res1=features["res1"],
            res2=features["res2"],
            scm2=features["scm2"],
            scm4=features["scm4"],
        )
        calibrated_feature = final_feature + delta_feature
        rgb_residual = self.feat_extract[5](calibrated_feature)
        outputs.append(rgb_residual + x)
        self._last_nopost_aux = aux
        if return_aux:
            return outputs, aux
        return outputs

    def nopost_regularization(self):
        aux = self._last_nopost_aux
        if aux is None:
            return {}
        return {
            "gate_mean": aux["g_low"].mean() + aux["g_detail"].mean(),
            "raw_action_abs_mean": aux["raw_action"].abs().mean(),
            "delta_feature_abs_mean": aux["delta_feature"].abs().mean(),
        }

    def nopost_stats(self):
        return summarize_aux(self._last_nopost_aux)


def build_net(
    version,
    data,
    fam_mode="original",
    nopost_use_low=True,
    nopost_use_detail=False,
    nopost_gate_bias=-3.0,
    nopost_context_channels=32,
    nopost_hidden_channels=32,
):
    if fam_mode != "original":
        raise ValueError("NoPostFGAConvIR keeps the official FAM path; use fam_mode='original'.")
    return NoPostFGAConvIR(
        version=version,
        data=data,
        nopost_use_low=nopost_use_low,
        nopost_use_detail=nopost_use_detail,
        nopost_gate_bias=nopost_gate_bias,
        nopost_context_channels=nopost_context_channels,
        nopost_hidden_channels=nopost_hidden_channels,
    )
