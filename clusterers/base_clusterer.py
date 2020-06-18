import copy

import torch
import numpy as np

class BaseClusterer():
    def __init__(self,
                 discriminator,
                 k_value=-1,
                 x_cluster=None,
                 batch_size=100,
                 **kwargs):
        ''' requires that self.x is not on the gpu, or else it hogs too much gpu memory ''' 
        self.cluster_counts = [0] * k_value
        self.discriminator = copy.deepcopy(discriminator)
        self.discriminator.eval()
        self.k = k_value
        self.kmeans = None
        self.x = x_cluster
        self.x_labels = None
        self.batch_size = batch_size

    def get_labels(self, x, y):
        return y

    def recluster(self, discriminator, **kwargs):
        return

    def get_features(self, x):
        ''' by default gets discriminator, but you can use other things '''
        return self.get_discriminator_output(x)

    def get_cluster_batch_features(self):
        ''' returns the discriminator features for the batch self.x as a numpy array '''
        with torch.no_grad():
            outputs = []
            x = self.x
            for batch in range(x.size(0) // self.batch_size):
                x_batch = x[batch * self.batch_size:(batch + 1) * self.batch_size].cuda()
                outputs.append(self.get_features(x_batch).detach().cpu())
            if (x.size(0) % self.batch_size != 0):
                x_batch = x[x.size(0) // self.batch_size * self.batch_size:].cuda()
                outputs.append(self.get_features(x_batch).detach().cpu())
            result = torch.cat(outputs, dim=0).numpy()
            return result

    def get_discriminator_output(self, x):
        '''returns discriminator features'''
        self.discriminator.eval()
        with torch.no_grad():
            return self.discriminator(x, get_features=True)

    def get_label_distribution(self, x=None):
        '''returns the empirical distributon of clustering'''
        y = self.x_labels if x is None else self.get_labels(x, None)
        counts = [0] * self.k
        for yi in y:
            counts[yi] += 1
        return counts

    def sample_y(self, batch_size):
        '''samples y according to the empirical distribution (not sure if used anymore)'''
        distribution = self.get_label_distribution()
        distribution = [i / sum(distribution) for i in distribution]
        m = torch.distributions.Multinomial(batch_size,
                                            torch.tensor(distribution))
        return m.sample()

    def print_label_distribution(self, x=None):
        print(self.get_label_distribution(x))
