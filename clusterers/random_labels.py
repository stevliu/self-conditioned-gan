import torch
from clusterers import base_clusterer


class Clusterer(base_clusterer.BaseClusterer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_labels(self, x, y):
        return torch.randint(low=0, high=self.k, size=y.shape).long().cuda()