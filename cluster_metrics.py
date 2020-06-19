import argparse
import os
from tqdm import tqdm

import torch
import numpy as np
from torch import nn

from gan_training import utils
from gan_training.inputs import get_dataset
from gan_training.checkpoints import CheckpointIO
from gan_training.config import load_config
from gan_training.metrics.clustering_metrics import (nmi, purity_score)

torch.backends.cudnn.benchmark = True

# Arguments
parser = argparse.ArgumentParser(description='Evaluate the clustering inferred by our method')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--model_it', type=str)
parser.add_argument('--random', action='store_true', help='Figure out if the clusters were randomly assigned')

args = parser.parse_args()
config = load_config(args.config, 'configs/default.yaml')
out_dir = config['training']['out_dir']


def main():
    checkpoint_dir = os.path.join(out_dir, 'chkpts')
    batch_size = config['training']['batch_size']

    if 'cifar' in config['data']['train_dir'].lower():
        name = 'cifar10'
    elif 'stacked_mnist' == config['data']['type']:
        name = 'stacked_mnist'
    else:
        name = 'image'

    if os.path.exists(os.path.join(out_dir, 'cluster_preds.npz')):
        # if we've already computed assignments, load them and move on
        with np.load(os.path.join(out_dir, 'cluster_preds.npz')) as f:
            y_reals = f['y_reals']
            y_preds = f['y_preds']
    else:
        train_dataset, _ = get_dataset(
            name=name,
            data_dir=config['data']['train_dir'],
            size=config['data']['img_size'])

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=config['training']['nworkers'],
            shuffle=True,
            pin_memory=True,
            sampler=None,
            drop_last=True)

        checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)

        print('Loading clusterer:')
        most_recent = utils.get_most_recent(checkpoint_dir, 'model') if args.model_it is None else args.model_it
        clusterer = checkpoint_io.load_clusterer(most_recent, load_samples=False, pretrained=config['pretrained'])

        if isinstance(clusterer.discriminator, nn.DataParallel):
            clusterer.discriminator = clusterer.discriminator.module

        y_preds = []
        y_reals = []

        for batch_num, (x_real, y_real) in enumerate(tqdm(train_loader, total=len(train_loader))):
            y_pred = clusterer.get_labels(x_real.cuda(), None)
            y_preds.append(y_pred.detach().cpu())
            y_reals.append(y_real)

        y_reals = torch.cat(y_reals).numpy()
        y_preds = torch.cat(y_preds).numpy()

        np.savez(os.path.join(out_dir, 'cluster_preds.npz'), y_reals=y_reals, y_preds=y_preds)

    if args.random:
        y_preds = np.random.randint(0, 100, size=y_reals.shape)

    nmi_score = nmi(y_preds, y_reals)
    purity = purity_score(y_preds, y_reals)
    print('nmi', nmi_score, 'purity', purity)


if __name__ == '__main__':
    main()
