import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed
from collections import OrderedDict

class Generator(nn.Module):
    '''
    Perfectly equivalent to resnet2.Generator (can load state dicts
    from that class), but organizes layers as a sequence for more
    automatic inversion.
    '''
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64,
            use_class_labels=False, **kwargs):
        super().__init__()
        s0 = self.s0 = size // 32
        nf = self.nf = nfilter
        self.z_dim = z_dim
        self.use_class_labels = use_class_labels

        # Submodules
        if use_class_labels:
            self.condition = ConditionGen(z_dim, nlabels, embed_size)
            latent_dim = self.condition.latent_dim
        else:
            latent_dim = z_dim

        self.layers = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(latent_dim, 16*nf*s0*s0)),
            ('reshape', Reshape(16*self.nf, self.s0, self.s0)),
            ('resnet_0_0', ResnetBlock(16*nf, 16*nf)),
            ('resnet_0_1', ResnetBlock(16*nf, 16*nf)),
            ('upsample_1', nn.Upsample(scale_factor=2)),
            ('resnet_1_0', ResnetBlock(16*nf, 16*nf)),
            ('resnet_1_1', ResnetBlock(16*nf, 16*nf)),
            ('upsample_2', nn.Upsample(scale_factor=2)),
            ('resnet_2_0', ResnetBlock(16*nf, 8*nf)),
            ('resnet_2_1', ResnetBlock(8*nf, 8*nf)),
            ('upsample_3', nn.Upsample(scale_factor=2)),
            ('resnet_3_0', ResnetBlock(8*nf, 4*nf)),
            ('resnet_3_1', ResnetBlock(4*nf, 4*nf)),
            ('upsample_4', nn.Upsample(scale_factor=2)),
            ('resnet_4_0', ResnetBlock(4*nf, 2*nf)),
            ('resnet_4_1', ResnetBlock(2*nf, 2*nf)),
            ('upsample_5', nn.Upsample(scale_factor=2)),
            ('resnet_5_0', ResnetBlock(2*nf, 1*nf)),
            ('resnet_5_1', ResnetBlock(1*nf, 1*nf)),
            ('img_relu', nn.LeakyReLU(2e-1)),
            ('conv_img', nn.Conv2d(nf, 3, 3, padding=1)),
            ('tanh', nn.Tanh())
        ]))

    def forward(self, z, y=None):
        assert(y is None or z.size(0) == y.size(0))
        assert(not self.use_class_labels or y is not None)
        batch_size = z.size(0)
        if self.use_class_labels:
            z = self.condition(z, y)
        return self.layers(z)

    def load_v2_state_dict(self, state_dict):
        converted = {}
        for k, v in state_dict.items():
            if k.startswith('embedding'):
                k = 'condition.' + k
            elif k == 'get_latent.embedding.weight':
                k = 'condition.embedding.weight'
            else:
                k = 'layers.' + k
            converted[k] = v
        self.load_state_dict(converted)

class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(*((batch_size,) + self.shape))

class ConditionGen(nn.Module):
    def __init__(self, z_dim, nlabels, embed_size=256):
        super().__init__()
        self.embedding = nn.Embedding(nlabels, embed_size)
        self.latent_dim = z_dim + embed_size
        self.z_dim = z_dim
        self.nlabels = nlabels
        self.embed_size = embed_size

    def forward(self, z, y):
        assert(z.size(0) == y.size(0))
        batch_size = z.size(0)
        if y.dtype is torch.int64:
            yembed = self.embedding(y)
        else:
            yembed = y
        yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)
        return torch.cat([z, yembed], dim=1)

def convert_from_resnet2_generator(gen):
    nlabels, embed_size = 0, 0
    
    if hasattr(gen, 'get_latent'):
        # new version does not have gen.use_class_labels..
        nlabels = gen.get_latent.embedding.num_embeddings
        embed_size = gen.get_latent.embedding.embedding_dim
        use_class_labels = True
    elif gen.use_class_labels:
        nlabels = gen.embedding.num_embeddings
        embed_size = gen.embedding.embedding_dim
        use_class_labels = True

    size = gen.s0 * 32
    newgen = Generator(gen.z_dim, nlabels, size, embed_size, gen.nf, use_class_labels)
    newgen.load_v2_state_dict(gen.state_dict())
    return newgen


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
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
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden,
                kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout,
                kernel_size=3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout,
                    kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

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