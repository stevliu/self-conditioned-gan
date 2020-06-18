import copy, random

import torch
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

from clusterers import kmeans


class Clusterer(kmeans.Clusterer):
    def __init__(self, initialization=True, matching=True, **kwargs):
        self.initialization = initialization
        self.matching = matching

        super().__init__(**kwargs)

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

    def fit_means(self):
        features = self.get_cluster_batch_features()

        # if clustered already, use old assignments for the cluster mean
        if self.x_labels is not None and self.initialization:
            print('Initializing k-means with previous cluster assignments')
            initialization = self.get_initialization(features, self.x_labels)
        else:
            initialization = 'k-means++'

        new_classes = self.kmeans_fit_predict(features, init=initialization)

        # we've clustered already, so compute the permutation
        if self.x_labels is not None and self.matching:
            print('Doing cluster matching')
            matching = self.hungarian_match(new_classes, self.x_labels, self.k,
                                            self.k)
            self.mapping = [int(j) for i, j in sorted(matching)]

        # recompute the fixed labels
        self.x_labels = np.array([self.mapping[x] for x in new_classes])

    def recluster(self, discriminator, **kwargs):
        self.discriminator = copy.deepcopy(discriminator)
        self.fit_means()

    def hungarian_match(self, flat_preds, flat_targets, preds_k, targets_k):
        '''takes in np arrays flat_preds, flat_targets of integers'''
        num_samples = flat_targets.shape[0]

        assert (preds_k == targets_k)  # one to one
        num_k = preds_k
        num_correct = np.zeros((num_k, num_k))

        for c1 in range(num_k):
            for c2 in range(num_k):
                votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
                num_correct[c1, c2] = votes

        # num_correct is small
        match = linear_assignment(num_samples - num_correct)

        # return as list of tuples, out_c to gt_c
        res = []
        for out_c, gt_c in match:
            res.append((out_c, gt_c))

        return res
