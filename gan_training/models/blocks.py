import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class ResnetBlock(nn.Module):
    def __init__(self,
                 fin,
                 fout,
                 bn,
                 nclasses,
                 fhidden=None,
                 is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden
        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden,
                      self.fout,
                      3,
                      stride=1,
                      padding=1,
                      bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin,
                          self.fout,
                          1,
                          stride=1,
                          padding=0,
                          bias=False)
        self.bn0 = bn(self.fin, nclasses)
        self.bn1 = bn(self.fhidden, nclasses)

    def forward(self, x, y):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(self.bn0(x, y)))
        dx = self.conv_1(actvn(self.bn1(dx, y)))
        out = x_s + 0.1 * dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out


class LatentEmbeddingConcat(nn.Module):
    ''' projects class embedding onto hypersphere and returns the concat of the latent and the class embedding '''

    def __init__(self, nlabels, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(nlabels, embed_dim)

    def forward(self, z, y):
        assert (y.size(0) == z.size(0))
        yembed = self.embedding(y)
        yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)
        yz = torch.cat([z, yembed], dim=1)
        return yz


class NormalizeLinear(nn.Module):
    def __init__(self, act_dim, k_value):
        super().__init__()
        self.lin = nn.Linear(act_dim, k_value)

    def normalize(self):
        self.lin.weight.data = F.normalize(self.lin.weight.data, p=2, dim=1)

    def forward(self, x):
        self.normalize()
        return self.lin(x)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, inp, *args, **kwargs):
        return inp


class LinearConditionalMaskLogits(nn.Module):
    ''' runs activated logits through fc and masks out the appropriate discriminator score according to class number'''

    def __init__(self, nc, nlabels):
        super().__init__()
        self.fc = nn.Linear(nc, nlabels)

    def forward(self, inp, y=None, take_best=False, get_features=False):
        out = self.fc(inp)
        if get_features: return out

        if not take_best:
            y = y.view(-1)
            index = Variable(torch.LongTensor(range(out.size(0))))
            if y.is_cuda:
                index = index.cuda()
            return out[index, y]
        else:
            # high activation means real, so take the highest activations
            best_logits, _ = out.max(dim=1)
            return best_logits


class ProjectionDiscriminatorLogits(nn.Module):
    ''' takes in activated flattened logits before last linear layer and implements https://arxiv.org/pdf/1802.05637.pdf '''

    def __init__(self, nc, nlabels):
        super().__init__()
        self.fc = nn.Linear(nc, 1)
        self.embedding = nn.Embedding(nlabels, nc)
        self.nlabels = nlabels

    def forward(self, x, y, take_best=False):
        output = self.fc(x)

        if not take_best:
            label_info = torch.sum(self.embedding(y) * x, dim=1, keepdim=True)
            return (output + label_info).view(x.size(0))
        else:
            #TODO: this may be computationally expensive, maybe we want to do the global pooling first to reduce x's size
            index = torch.LongTensor(range(self.nlabels)).cuda()
            labels = index.repeat((x.size(0), ))
            x = x.repeat_interleave(self.nlabels, dim=0)
            label_info = torch.sum(self.embedding(labels) * x,
                                   dim=1,
                                   keepdim=True).view(output.size(0),
                                                      self.nlabels)
            # high activation means real, so take the highest activations
            best_logits, _ = label_info.max(dim=1)
            return output.view(output.size(0)) + best_logits


class LinearUnconditionalLogits(nn.Module):
    ''' standard discriminator logit layer '''

    def __init__(self, nc):
        super().__init__()
        self.fc = nn.Linear(nc, 1)

    def forward(self, inp, y, take_best=False):
        assert (take_best == False)

        out = self.fc(inp)
        return out.view(out.size(0))


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(*((batch_size, ) + self.shape))


class ConditionalBatchNorm2d(nn.Module):
    ''' from https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775 '''

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(
            1, 0.02)  # Initialize scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_(
        )  # Initialize bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(
            -1, self.num_features, 1, 1)
        return out


class BatchNorm2d(nn.Module):
    ''' identical to nn.BatchNorm2d but takes in y input that is ignored '''

    def __init__(self, nc, nchannels, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm2d(nc)

    def forward(self, x, y):
        return self.bn(x)
