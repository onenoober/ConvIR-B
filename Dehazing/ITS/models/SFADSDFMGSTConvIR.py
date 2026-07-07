import torch
import torch.nn.functional as F

from .ConvIR import ConvIR
from .SFADConvIR import (
    SpatialDegradationFieldModulation,
    configure_sfad_train_scope,
    load_haze4k_partial,
)
from .SFADGSTConvIR import GuidedSkipTransfer


class SFADSDFMGSTConvIR(ConvIR):
    def __init__(self, version, data):
        super().__init__(version, data)
        base_channel = 32
        self.SFAD_SDFM1 = SpatialDegradationFieldModulation(base_channel * 4)
        self.SFAD_SDFM2 = SpatialDegradationFieldModulation(base_channel * 2)
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
        z = self.SFAD_SDFM2(x_2, z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.SFAD_SDFM1(x_4, z, z4)
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
            'SDFM_1_4': self.SFAD_SDFM1.collect_stats(),
            'SDFM_1_2': self.SFAD_SDFM2.collect_stats(),
            'GST_1_2': self.SFAD_GST1.collect_stats(),
            'GST_1_1': self.SFAD_GST2.collect_stats(),
        }


def build_sfad_sdfm_gst_net(version, data, fam_mode='original'):
    if fam_mode != 'original':
        raise ValueError("SFAD-SDFM-GST route starts from fam_mode='original'.")
    return SFADSDFMGSTConvIR(version, data)
