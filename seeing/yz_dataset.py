import torch, numpy


class YZDataset():
    def __init__(self, zdim=256, nlabels=1, distribution=[1.], device='cpu'):
        self.zdim = zdim
        self.nlabels = nlabels
        self.device = device
        self.distribution = distribution
        assert (len(distribution) == nlabels)

    def __call__(self, seeds):
        zs, ys = [], []
        for seed in seeds:
            rng = numpy.random.RandomState(seed)
            z = torch.from_numpy(
                rng.standard_normal(self.zdim).reshape(
                    1, self.zdim)).float().to(self.device)
            y = torch.from_numpy(
                rng.choice(self.nlabels, 1, replace=False,
                           p=self.distribution)).long().to(self.device)
            zs.append(z)
            ys.append(y)
        return torch.cat(zs, dim=0), torch.cat(ys, dim=0)


if __name__ == '__main__':
    sampler = YZDataset()
    a, d = sampler([10, 11])
    b, e = sampler([12, 13])
    assert ((a - b).mean() > 1e-3)
    c, f = sampler([10, 11])
    assert ((a - c).mean() < 1e-3)
