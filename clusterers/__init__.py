from clusterers import (base_clusterer, selfcondgan, random_labels, online)

clusterer_dict = {
    'supervised': base_clusterer.BaseClusterer,
    'selfcondgan': selfcondgan.Clusterer,
    'online': online.Clusterer,
    'random_labels': random_labels.Clusterer
}
