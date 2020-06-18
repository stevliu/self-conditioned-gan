import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from gan_training.models import blocks


class Generator(nn.Module):
    def __init__(self,
                 nlabels,
                 conditioning,
                 z_dim=128,
                 nc=3,
                 ngf=64,
                 embed_dim=256,
                 **kwargs):
        super(Generator, self).__init__()

        assert conditioning != 'unconditional' or nlabels == 1

        if conditioning == 'embedding':
            self.get_latent = blocks.LatentEmbeddingConcat(nlabels, embed_dim)
            self.fc = nn.Linear(z_dim + embed_dim, 4 * 4 * ngf * 8)
        elif conditioning == 'unconditional':
            self.get_latent = blocks.Identity()
            self.fc = nn.Linear(z_dim, 4 * 4 * ngf * 8)
        else:
            raise NotImplementedError(
                f"{conditioning} not implemented for generator")

        bn = blocks.BatchNorm2d

        self.nlabels = nlabels

        self.conv1 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1)
        self.bn1 = bn(ngf * 4, nlabels)

        self.conv2 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1)
        self.bn2 = bn(ngf * 2, nlabels)

        self.conv3 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1)
        self.bn3 = bn(ngf, nlabels)

        self.conv_out = nn.Sequential(nn.Conv2d(ngf, nc, 3, 1, 1), nn.Tanh())

    def forward(self, input, y):
        y = y.clamp(None, self.nlabels - 1)

        out = self.get_latent(input, y)
        out = self.fc(out)

        out = out.view(out.size(0), -1, 4, 4)
        out = F.relu(self.bn1(self.conv1(out), y))
        out = F.relu(self.bn2(self.conv2(out), y))
        out = F.relu(self.bn3(self.conv3(out), y))
        return self.conv_out(out)


class Discriminator(nn.Module):
    def __init__(self,
                 nlabels,
                 conditioning,
                 features='penultimate',
                 pack_size=1,
                 nc=3,
                 ndf=64,
                 **kwargs):
        super(Discriminator, self).__init__()

        assert conditioning != 'unconditional' or nlabels == 1

        self.nlabels = nlabels

        self.conv1 = nn.Sequential(nn.Conv2d(nc * pack_size, ndf, 4, 2, 1),
                                   nn.BatchNorm2d(ndf),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
                                   nn.BatchNorm2d(ndf * 2),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
                                   nn.BatchNorm2d(ndf * 4),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
                                   nn.BatchNorm2d(ndf * 8),
                                   nn.LeakyReLU(0.2, inplace=True))

        if conditioning == 'mask':
            self.fc_out = blocks.LinearConditionalMaskLogits(ndf * 8 * 4 , nlabels)
        elif conditioning == 'unconditional':
            self.fc_out = blocks.LinearUnconditionalLogits(ndf * 8 * 4)
        else:
            raise NotImplementedError(
                f"{conditioning} not implemented for discriminator")

        self.pack_size = pack_size
        self.features = features
        print(f'Getting features from {self.features}')

    def stack(self, x):
        #pacgan
        nc = self.pack_size
        if nc == 1:
            return x
        x_new = []
        for i in range(x.size(0) // nc):
            imgs_to_stack = x[i * nc:(i + 1) * nc]
            x_new.append(torch.cat([t for t in imgs_to_stack], dim=0))
        return torch.stack(x_new)

    def forward(self, input, y=None, get_features=False):
        input = self.stack(input)
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.size(0), -1)
        if get_features: return out.view(out.size(0), -1)
        y = y.clamp(None, self.nlabels - 1)
        result = self.fc_out(out, y)
        assert (len(result.shape) == 1)
        return result


if __name__ == '__main__':
    z = torch.zeros((1, 128))
    g = Generator()
    x = torch.zeros((1, 3, 32, 32))
    d = Discriminator()

    g(z)
    d(g(z))
    d(x)
