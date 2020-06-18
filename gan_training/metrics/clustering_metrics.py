def warn(*args, **kwargs):
    pass


import warnings
warnings.warn = warn

from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score
from sklearn import metrics

import numpy as np


def nmi(inferred, gt):
    return normalized_mutual_info_score(inferred, gt)


def acc(inferred, gt):
    gt = gt.astype(np.int64)
    assert inferred.size == gt.size
    D = max(inferred.max(), gt.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(inferred.size):
        w[inferred[i], gt[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / inferred.size


def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix,
                          axis=0)) / np.sum(contingency_matrix)


def ari(inferred, gt):
    return adjusted_rand_score(gt, inferred)


def homogeneity(inferred, gt):
    return homogeneity_score(gt, inferred)
