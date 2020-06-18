def warn(*args, **kwargs):
    pass


import warnings
warnings.warn = warn

import numpy as np


def percent_good_grid(x_fake, var=0.0025, nrows=5, ncols=5):
    std = np.sqrt(var)
    x = list(range(nrows))
    y = list(range(ncols))

    threshold = 3 * std
    means = []
    for i in x:
        for j in y:
            means.append(np.array([x[i] * 2 - 4, y[j] * 2 - 4]))
    return percent_good_pts(x_fake, means, threshold)


def percent_good_ring(x_fake, var=0.0001, n_clusters=8, radius=2.0):
    std = np.sqrt(var)
    thetas = np.linspace(0, 2 * np.pi, n_clusters + 1)[:n_clusters]
    x, y = radius * np.sin(thetas), radius * np.cos(thetas)
    threshold = np.array([std * 3, std * 3])
    means = []
    for i in range(n_clusters):
        means.append(np.array([x[i], y[i]]))
    return percent_good_pts(x_fake, means, threshold)


def percent_good_pts(x_fake, means, threshold):
    """Calculate %good, #modes, kl

    Keyword arguments:
    x_fake -- detached generated samples
    means -- true means
    threshold -- good point if l_1 distance is within threshold
    """
    count = 0
    counts = np.zeros(len(means))
    visited = set()
    for point in x_fake:
        minimum = 0
        diff_minimum = [1e10, 1e10]
        for i, mean in enumerate(means):
            diff = np.abs(point - mean)
            if np.all(diff < threshold):
                visited.add(tuple(mean))
                count += 1
                break
        for i, mean in enumerate(means):
            diff = np.abs(point - mean)
            if np.linalg.norm(diff) < np.linalg.norm(diff_minimum):
                minimum = i
                diff_minimum = diff
        counts[minimum] += 1

    kl = 0
    counts = counts / len(x_fake)
    for generated in counts:
        if generated != 0:
            kl += generated * np.log(len(means) * generated)

    return count / len(x_fake), len(visited), kl
