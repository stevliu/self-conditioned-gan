import argparse
import os
import shutil
import torch
import torchvision

from torch import nn
from gan_training import utils
from gan_training.inputs import get_dataset
from gan_training.checkpoints import CheckpointIO
from gan_training.config import load_config
from seeded_sampler import SeededSampler

torch.backends.cudnn.benchmark = True

# Arguments
parser = argparse.ArgumentParser(description='Visualize the samples/clusters of a class-conditional GAN')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--model_it', type=int, help='If you want to load from a specific model iteration')
parser.add_argument('--show_clusters', action='store_true', help='show the real images. Requires a path to the real image train directory')
args = parser.parse_args()

config = load_config(args.config, 'configs/default.yaml')
out_dir = config['training']['out_dir']


def main():
    checkpoint_dir = os.path.join(out_dir, 'chkpts')

    most_recent = utils.get_most_recent(checkpoint_dir, 'model') if args.model_it is None else args.model_it

    cluster_path = os.path.join(out_dir, 'clusters')
    print('Saving clusters/samples to', cluster_path)

    os.makedirs(cluster_path, exist_ok=True)

    shutil.copyfile('seeing/lightbox.html', os.path.join(cluster_path, '+lightbox.html'))

    checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)

    most_recent = utils.get_most_recent(checkpoint_dir, 'model') if args.model_it is None else args.model_it
    clusterer = checkpoint_io.load_clusterer(most_recent, pretrained=config['pretrained'], load_samples=False)

    if isinstance(clusterer.discriminator, nn.DataParallel):
        clusterer.discriminator = clusterer.discriminator.module

    model_path = os.path.join(checkpoint_dir, 'model_%08d.pt' % most_recent)
    sampler = SeededSampler(args.config,
                            model_path=model_path,
                            clusterer_path=os.path.join(checkpoint_dir, f'clusterer{most_recent}.pkl'),
                            pretrained=config['pretrained'])

    if args.show_clusters:
        clusters = [[] for _ in range(config['generator']['nlabels'])]
        train_dataset, _ = get_dataset(
            name='webp'
            if 'cifar' not in config['data']['train_dir'].lower() else 'cifar10',
            data_dir=config['data']['train_dir'],
            size=config['data']['img_size'])

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['nworkers'],
            shuffle=True,
            pin_memory=True,
            sampler=None,
            drop_last=True)

        print('Generating clusters')
        for batch_num, (x_real, y_gt) in enumerate(train_loader):
            x_real = x_real.cuda()
            y_pred = clusterer.get_labels(x_real, y_gt)

            for i, yi in enumerate(y_pred):
                clusters[yi].append(x_real[i].cpu())

            # don't generate too many, we're only visualizing 20 per cluster
            if batch_num * config['training']['batch_size'] >= 10000:
                break
    else:
        clusters = [None] * config['generator']['nlabels']

    nimgs = 20
    nrows = 4

    for i in range(len(clusters)):
        if clusters[i] is None:
            pass
        elif len(clusters[i]) >= nimgs:
            cluster = torch.stack(clusters[i])[:nimgs]

            torchvision.utils.save_image(cluster * 0.5 + 0.5,
                                         os.path.join(cluster_path, f'{i}_real.png'),
                                         nrow=nrows)
        generated = []
        for seed in range(nimgs):
            img = sampler.conditional_sample(i, seed=seed)
            generated.append(img.detach().cpu())
        generated = torch.cat(generated)

        torchvision.utils.save_image(generated * 0.5 + 0.5,
                                     os.path.join(cluster_path, f'{i}_gen.png'),
                                     nrow=nrows)

    print('Clusters/samples can be visualized under', cluster_path)


if __name__ == '__main__':
    main()
