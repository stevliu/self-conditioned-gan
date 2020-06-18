import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.append('../gan_training/models')

from blocks import LatentEmbeddingConcat, Identity, LinearUnconditionalLogits, LinearConditionalMaskLogits


class G(nn.Module):
    def __init__(self,
                 conditioning,
                 k_value,
                 z_dim=2,
                 embed_size=32,
                 act_dim=400,
                 x_dim=2):

        super().__init__()
        if conditioning == 'unconditional':
            embed_size = 0
            self.embedding = Identity()
        elif conditioning == 'conditional':
            self.embedding = LatentEmbeddingConcat(k_value, embed_size)
        else:
            raise NotImplementedError()

        self.fc1 = nn.Sequential(nn.Linear(z_dim + embed_size, act_dim),
                                 nn.BatchNorm1d(act_dim), nn.ReLU(True))
        self.fc2 = nn.Sequential(nn.Linear(act_dim, act_dim),
                                 nn.BatchNorm1d(act_dim), nn.ReLU(True))
        self.fc3 = nn.Sequential(nn.Linear(act_dim, act_dim),
                                 nn.BatchNorm1d(act_dim), nn.ReLU(True))
        self.fc4 = nn.Sequential(nn.Linear(act_dim, act_dim),
                                 nn.BatchNorm1d(act_dim), nn.ReLU(True))
        self.fc_out = nn.Linear(act_dim, x_dim)

    def forward(self, z, y=None):
        out = self.fc1(self.embedding(z, y))
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc_out(out)
        return out


class D(nn.Module):
    class Maxout(nn.Module):
        # Taken from https://github.com/pytorch/pytorch/issues/805
        def __init__(self, d_in, d_out, pool_size=5):
            super().__init__()
            self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
            self.lin = nn.Linear(d_in, d_out * pool_size)

        def forward(self, inputs):
            shape = list(inputs.size())
            shape[-1] = self.d_out
            shape.append(self.pool_size)
            max_dim = len(shape) - 1
            out = self.lin(inputs)
            m, i = out.view(*shape).max(max_dim)
            return m

    def max(self, out, dim=5):
        return out.view(out.size(0), -1, dim).max(2)[0]

    def __init__(self, conditioning, k_value, act_dim=200, x_dim=2):
        super().__init__()
        self.fc1 = self.Maxout(x_dim, act_dim)
        self.fc2 = self.Maxout(act_dim, act_dim)
        self.fc3 = self.Maxout(act_dim, act_dim)

        if conditioning == 'unconditional':
            self.fc4 = LinearUnconditionalLogits(act_dim)
        elif conditioning == 'conditional':
            self.fc4 = LinearConditionalMaskLogits(act_dim, k_value)
        else:
            raise NotImplementedError()

    def forward(self, x, y=None, get_features=False):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        if get_features: return out
        return self.fc4(out, y, get_features=get_features)
