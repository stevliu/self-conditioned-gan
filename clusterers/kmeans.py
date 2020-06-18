import torch
import numpy as np
from sklearn.cluster import KMeans

from clusterers import base_clusterer

class Clusterer(base_clusterer.BaseClusterer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mapping = list(range(self.k))

    def kmeans_fit_predict(self, features, init='k-means++', n_init=10):
        '''fits kmeans, and returns the predictions of the kmeans'''
        print('Fitting k-means w data shape', features.shape)
        self.kmeans = KMeans(init=init, n_clusters=self.k,
                             n_init=n_init).fit(features)
        return self.kmeans.predict(features)

    def get_labels(self, x, y):
        d_features = self.get_features(x).detach().cpu().numpy()
        np_prediction = self.kmeans.predict(d_features)
        permuted_prediction = np.array([self.mapping[x] for x in np_prediction])
        return torch.from_numpy(permuted_prediction).long().cuda()
