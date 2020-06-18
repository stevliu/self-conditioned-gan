import copy, random

import torch
import numpy as np

from clusterers import kmeans


class Clusterer(kmeans.Clusterer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.burned_in = False

    def get_initialization(self, features, labels):
        '''given points (from new discriminator) and their old assignments as np arrays, compute the induced means as a np array'''
        means = []
        for i in range(self.k):
            mask = (labels == i)
            mean = np.zeros(features[0].shape)
            numels = mask.astype(int).sum()
            if numels > 0:
                for index, equal in enumerate(mask):
                    if equal: mean += features[index]
                means.append(mean / numels)
            else:
                # use kmeans++ init if cluster is starved
                rand_point = random.randint(0, features.size(0) - 1)
                means.append(features[rand_point])
        result = np.array(means)
        return result

    def recluster(self, discriminator, x_batch=None, **kwargs):
        if self.kmeans is None:
            print('kmeans clustering as initialization')
            self.discriminator = copy.deepcopy(discriminator)
            features = self.get_cluster_batch_features()
            self.x_labels = self.kmeans_fit_predict(features)
        else:
            self.discriminator = discriminator
            if not self.burned_in:
                print('Burned in: computing initialization for kmeans')
                features = self.get_cluster_batch_features()
                initialization = self.get_initialization(
                    features, self.x_labels)
                self.kmeans_fit_predict(features, init=initialization)
                self.burned_in = True
            else:
                assert x_batch is not None
                self.discriminator = discriminator
                features = self.get_features(x_batch).detach().cpu().numpy()
                y_pred = self.kmeans.predict(features)

                for xi, yi in zip(features, y_pred):
                    self.cluster_counts[yi] += 1
                    difference = xi - self.kmeans.cluster_centers_[yi]
                    step_size = 1.0 / self.cluster_counts[yi]
                    self.kmeans.cluster_centers_[
                        yi] = self.kmeans.cluster_centers_[yi] + step_size * (
                            difference)
