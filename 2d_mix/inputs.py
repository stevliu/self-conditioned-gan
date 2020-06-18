import numpy as np
import random

mapping = list(range(25))

def map_labels(labels):
    return np.array([mapping[label] for label in labels])


def get_data_ring(batch_size, radius=2.0, var=0.0001, n_clusters=8):
    thetas = np.linspace(0, 2 * np.pi, n_clusters + 1)[:n_clusters]
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
    classes = np.random.multinomial(batch_size,
                                    [1.0 / n_clusters] * n_clusters)
    labels = [i for i in range(n_clusters) for _ in range(classes[i])]
    random.shuffle(labels)
    labels = np.array(labels)
    samples = np.array([
        np.random.multivariate_normal([xs[i], ys[i]], [[var, 0], [0, var]])
        for i in labels
    ])
    return samples, labels


def get_data_grid(batch_size, radius=2.0, var=0.0025, nrows=5, ncols=5):
    samples = []
    labels = []
    for _ in range(batch_size):
        i, j = random.randint(0, ncols - 1), random.randint(0, nrows - 1)
        samples.append(
            np.random.multivariate_normal([i * 2 - 4, j * 2 - 4],
                                          [[var, 0], [0, var]]))
        labels.append(5 * i + j)
    return np.array(samples), map_labels(labels)
