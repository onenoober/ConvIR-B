import torch
import torch.nn as nn
import torch.nn.functional as F

from .ConvIR import ConvIR


class A0ProxResidualConvIR(ConvIR):
    """ConvIR-B plus a zero-init bounded residual head.

    The official ConvIR path is preserved exactly at initialization. The new
    head can later learn a small supervised correction, but Stage-0 must prove
    exact no-op behavior after partial-loading the official checkpoint.
    """

    def __init__(self, version, data, beta=0.05):
        super(A0ProxResidualConvIR, self).__init__(version, data)
        base_channel = 32
        self.register_buffer("A0PROX_beta", torch.tensor(float(beta)))
        self.A0PROX_head = nn.Sequential(
            nn.Conv2d(base_channel + 6, base_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channel, 3, kernel_size=3, padding=1),
        )
        nn.init.kaiming_normal_(self.A0PROX_head[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.A0PROX_head[0].bias)
        nn.init.zeros_(self.A0PROX_head[2].weight)
        nn.init.zeros_(self.A0PROX_head[2].bias)

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

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_ + x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        decoder_feature = z
        base_delta = self.feat_extract[5](decoder_feature)
        a0 = base_delta + x
        residual_input = torch.cat([decoder_feature, a0, x], dim=1)
        correction = self.A0PROX_beta * torch.tanh(self.A0PROX_head(residual_input))
        outputs.append(a0 + correction)

        return outputs


def build_a0prox_residual_net(version, data, beta=0.05):
    return A0ProxResidualConvIR(version, data, beta=beta)
