import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed

from gan_training.models import blocks
from gan_training.models.blocks import ResnetBlock
from torch.nn.utils.spectral_norm import spectral_norm


class Generator(nn.Module):
    def __init__(self,
                 z_dim,
                 nlabels,
                 size,
                 conditioning,
                 embed_size=256,
                 nfilter=64,
                 **kwargs):
        super().__init__()
        s0 = self.s0 = size // 32
        nf = self.nf = nfilter
        self.nlabels = nlabels
        self.z_dim = z_dim

        assert conditioning != 'unconditional' or nlabels == 1

        if conditioning == 'embedding':
            self.get_latent = blocks.LatentEmbeddingConcat(nlabels, embed_size)
            self.fc = nn.Linear(z_dim + embed_size, 16 * nf * s0 * s0)
        elif conditioning == 'unconditional':
            self.get_latent = blocks.Identity()
            self.fc = nn.Linear(z_dim, 16 * nf * s0 * s0)
        else:
            raise NotImplementedError(
                f"{conditioning} not implemented for generator")

        #either use conditional batch norm, or use no batch norm
        bn = blocks.Identity

        self.resnet_0_0 = ResnetBlock(16 * nf, 16 * nf, bn, nlabels)
        self.resnet_0_1 = ResnetBlock(16 * nf, 16 * nf, bn, nlabels)

        self.resnet_1_0 = ResnetBlock(16 * nf, 16 * nf, bn, nlabels)
        self.resnet_1_1 = ResnetBlock(16 * nf, 16 * nf, bn, nlabels)

        self.resnet_2_0 = ResnetBlock(16 * nf, 8 * nf, bn, nlabels)
        self.resnet_2_1 = ResnetBlock(8 * nf, 8 * nf, bn, nlabels)

        self.resnet_3_0 = ResnetBlock(8 * nf, 4 * nf, bn, nlabels)
        self.resnet_3_1 = ResnetBlock(4 * nf, 4 * nf, bn, nlabels)

        self.resnet_4_0 = ResnetBlock(4 * nf, 2 * nf, bn, nlabels)
        self.resnet_4_1 = ResnetBlock(2 * nf, 2 * nf, bn, nlabels)

        self.resnet_5_0 = ResnetBlock(2 * nf, 1 * nf, bn, nlabels)
        self.resnet_5_1 = ResnetBlock(1 * nf, 1 * nf, bn, nlabels)

        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, z, y):
        y = y.clamp(None, self.nlabels - 1)
        out = self.get_latent(z, y)

        out = self.fc(out)

        out = out.view(z.size(0), 16 * self.nf, self.s0, self.s0)

        out = self.resnet_0_0(out, y)
        out = self.resnet_0_1(out, y)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_1_0(out, y)
        out = self.resnet_1_1(out, y)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_2_0(out, y)
        out = self.resnet_2_1(out, y)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_3_0(out, y)
        out = self.resnet_3_1(out, y)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_4_0(out, y)
        out = self.resnet_4_1(out, y)

        out = F.interpolate(out, scale_factor=2)
        out = self.resnet_5_0(out, y)
        out = self.resnet_5_1(out, y)

        out = self.conv_img(actvn(out))
        out = torch.tanh(out)

        return out


class Discriminator(nn.Module):
    def __init__(self,
                 nlabels,
                 size,
                 conditioning,
                 nfilter=64,
                 features='penultimate',
                 **kwargs):
        super().__init__()
        s0 = self.s0 = size // 32
        nf = self.nf = nfilter
        self.nlabels = nlabels

        assert conditioning != 'unconditional' or nlabels == 1
        bn = blocks.Identity

        self.conv_img = nn.Conv2d(3, 1 * nf, 3, padding=1)

        self.resnet_0_0 = ResnetBlock(1 * nf, 1 * nf, bn, nlabels)
        self.resnet_0_1 = ResnetBlock(1 * nf, 2 * nf, bn, nlabels)

        self.resnet_1_0 = ResnetBlock(2 * nf, 2 * nf, bn, nlabels)
        self.resnet_1_1 = ResnetBlock(2 * nf, 4 * nf, bn, nlabels)

        self.resnet_2_0 = ResnetBlock(4 * nf, 4 * nf, bn, nlabels)
        self.resnet_2_1 = ResnetBlock(4 * nf, 8 * nf, bn, nlabels)

        self.resnet_3_0 = ResnetBlock(8 * nf, 8 * nf, bn, nlabels)
        self.resnet_3_1 = ResnetBlock(8 * nf, 16 * nf, bn, nlabels)

        self.resnet_4_0 = ResnetBlock(16 * nf, 16 * nf, bn, nlabels)
        self.resnet_4_1 = ResnetBlock(16 * nf, 16 * nf, bn, nlabels)

        self.resnet_5_0 = ResnetBlock(16 * nf, 16 * nf, bn, nlabels)
        self.resnet_5_1 = ResnetBlock(16 * nf, 16 * nf, bn, nlabels)

        if conditioning == 'mask':
            self.fc_out = blocks.LinearConditionalMaskLogits(
                16 * nf * s0 * s0, nlabels)
        elif conditioning == 'unconditional':
            self.fc_out = blocks.LinearUnconditionalLogits(16 * nf * s0 * s0)
        else:
            raise NotImplementedError(
                f"{conditioning} not implemented for discriminator")

        self.features = features

    def forward(self, x, y=None, get_features=False):
        batch_size = x.size(0)
        if y is not None:
            y = y.clamp(None, self.nlabels - 1)

        out = self.conv_img(x)

        out = self.resnet_0_0(out, y)
        out = self.resnet_0_1(out, y)
        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_1_0(out, y)
        out = self.resnet_1_1(out, y)
        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_2_0(out, y)
        out = self.resnet_2_1(out, y)
        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_3_0(out, y)
        out = self.resnet_3_1(out, y)
        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_4_0(out, y)
        out = self.resnet_4_1(out, y)
        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_5_0(out, y)
        out = self.resnet_5_1(out, y)
        out = actvn(out)

        if get_features and self.features == 'summed':
            return out.view(out.size(0), out.size(1), -1).sum(dim=2)

        out = out.view(batch_size, 16 * self.nf * self.s0 * self.s0)

        if get_features: return out.view(batch_size, -1)
        result = self.fc_out(out, y)
        assert (len(result.shape) == 1)
        return result


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out