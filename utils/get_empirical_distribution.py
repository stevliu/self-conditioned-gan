import argparse
import os
from tqdm import tqdm

import json
import numpy as np

from classifiers import classifier_dict
from np_to_pt_img import np_to_pt


def get_empirical_distribution(path_to_samples):
    ''' gets the fake and real distributions induced by the classifier '''
    results = {}

    with np.load(path_to_samples, allow_pickle=True) as data:
        for datatype in ['fake']:  # , 'real'
            counts = {}
            results[datatype] = counts
            imgs = data[datatype]
            print(f'Found {len(imgs)} samples in {path_to_samples}')
            for it in tqdm(range(len(imgs) // batch_size)):
                x_batch = np_to_pt(imgs[it * batch_size:(it + 1) * batch_size]).cuda()
                y_pred = classifier.get_predictions(x_batch)
                for yi in y_pred:
                    yi = yi.item()
                    if yi not in counts:
                        counts[yi] = 0
                    counts[yi] += 1
            counts = {str(k): v / len(imgs) for k, v in counts.items()}
    return results


def get_kl(fake, nclasses):
    '''computes the log10 kl between empirical distributions.'''
    result = 0
    total = sum([v for k, v in fake.items()])
    for c, count in fake.items():
        pi = count / total
        # log10 seems to reproduce pacgan results
        result += pi * np.log10(pi * nclasses)
    return result


nmodes_gt = {'places': 365, 'cifar': 10, 'imagenet': 1000, 'stacked_mnist': 1000}

if __name__ == '__main__':
    parser = argparse.ArgumentParser('compute empirical distributions and reverse-kl metrics')
    parser.add_argument('--samples', help='path to samples')
    parser.add_argument('--it', type=str, help='iteration number (can be \'pretrained\') of samples')
    parser.add_argument('--results_dir', help='path to results_dir')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=100)
    args = parser.parse_args()

    batch_size = args.batch_size
    classifier = classifier_dict[args.dataset]()
    it = args.it
    results_dir = args.results_dir
    result = get_empirical_distribution(args.samples)
    nmodes = len(result['fake'])
    nclasses = nmodes_gt[args.dataset]

    kl = get_kl(result['fake'], nclasses)

    with open(os.path.join(args.results_dir, 'kl_results.json')) as f:
        kl_results = json.load(f)
    with open(os.path.join(args.results_dir, 'nmodes_results.json')) as f:
        nmodes_results = json.load(f)

    kl_results[it] = kl
    nmodes_results[it] = nmodes

    print(f'{results_dir} iteration {it} KL: {kl} Covered {nmodes} out of {nclasses} total modes')

    with open(os.path.join(args.results_dir, 'kl_results.json'), 'w') as f:
        f.write(json.dumps(kl_results))
    with open(os.path.join(args.results_dir, 'nmodes_results.json'), 'w') as f:
        f.write(json.dumps(nmodes_results))
