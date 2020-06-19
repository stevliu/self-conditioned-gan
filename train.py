import argparse
import os
import copy
import pprint
from os import path

import torch
import numpy as np
from torch import nn

from gan_training import utils
from gan_training.train import Trainer, update_average
from gan_training.logger import Logger
from gan_training.checkpoints import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (load_config, get_clusterer, build_models, build_optimizers)
from seeing.pidfile import exit_if_job_done, mark_job_done

torch.backends.cudnn.benchmark = True

# Arguments
parser = argparse.ArgumentParser(
    description='Train a GAN with different regularization strategies.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--outdir', type=str, help='used to override outdir (useful for multiple runs)')
parser.add_argument('--nepochs', type=int, default=250, help='number of epochs to run before terminating')
parser.add_argument('--model_it', type=int, default=-1, help='which model iteration to load from, -1 loads the most recent model')
parser.add_argument('--devices', nargs='+', type=str, default=['0'], help='devices to use')

args = parser.parse_args()
config = load_config(args.config, 'configs/default.yaml')
out_dir = config['training']['out_dir'] if args.outdir is None else args.outdir


def main():
    pp = pprint.PrettyPrinter(indent=1)
    pp.pprint({
        'data': config['data'],
        'generator': config['generator'],
        'discriminator': config['discriminator'],
        'clusterer': config['clusterer'],
        'training': config['training']
    })
    is_cuda = torch.cuda.is_available()

    # Short hands
    batch_size = config['training']['batch_size']
    log_every = config['training']['log_every']
    inception_every = config['training']['inception_every']
    backup_every = config['training']['backup_every']
    sample_nlabels = config['training']['sample_nlabels']
    nlabels = config['data']['nlabels']
    sample_nlabels = min(nlabels, sample_nlabels)

    checkpoint_dir = path.join(out_dir, 'chkpts')

    # Create missing directories
    if not path.exists(out_dir):
        os.makedirs(out_dir)
    if not path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Logger
    checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)

    device = torch.device("cuda:0" if is_cuda else "cpu")

    train_dataset, _ = get_dataset(
        name=config['data']['type'],
        data_dir=config['data']['train_dir'],
        size=config['data']['img_size'],
        deterministic=config['data']['deterministic'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config['training']['nworkers'],
        shuffle=True,
        pin_memory=True,
        sampler=None,
        drop_last=True)

    # Create models
    generator, discriminator = build_models(config)

    # Put models on gpu if needed
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    for name, module in discriminator.named_modules():
        if isinstance(module, nn.Sigmoid):
            print('Found sigmoid layer in discriminator; not compatible with BCE with logits')
            exit()

    g_optimizer, d_optimizer = build_optimizers(generator, discriminator, config)

    devices = [int(x) for x in args.devices]
    generator = nn.DataParallel(generator, device_ids=devices)
    discriminator = nn.DataParallel(discriminator, device_ids=devices)

    # Register modules to checkpoint
    checkpoint_io.register_modules(generator=generator,
                                   discriminator=discriminator,
                                   g_optimizer=g_optimizer,
                                   d_optimizer=d_optimizer)

    # Logger
    logger = Logger(log_dir=path.join(out_dir, 'logs'),
                    img_dir=path.join(out_dir, 'imgs'),
                    monitoring=config['training']['monitoring'],
                    monitoring_dir=path.join(out_dir, 'monitoring'))

    # Distributions
    ydist = get_ydist(nlabels, device=device)
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)

    ntest = config['training']['ntest']
    x_test, y_test = utils.get_nsamples(train_loader, ntest)
    x_cluster, y_cluster = utils.get_nsamples(train_loader, config['clusterer']['nimgs'])
    x_test, y_test = x_test.to(device), y_test.to(device)
    z_test = zdist.sample((ntest, ))
    utils.save_images(x_test, path.join(out_dir, 'real.png'))
    logger.add_imgs(x_test, 'gt', 0)

    # Test generator
    if config['training']['take_model_average']:
        print('Taking model average')
        bad_modules = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
        for model in [generator, discriminator]:
            for name, module in model.named_modules():
                for bad_module in bad_modules:
                    if isinstance(module, bad_module):
                        print('Batch norm in discriminator not compatible with exponential moving average')
                        exit()
        generator_test = copy.deepcopy(generator)
        checkpoint_io.register_modules(generator_test=generator_test)
    else:
        generator_test = generator

    clusterer = get_clusterer(config)(discriminator=discriminator,
                                      x_cluster=x_cluster,
                                      x_labels=y_cluster,
                                      gt_nlabels=config['data']['nlabels'],
                                      **config['clusterer']['kwargs'])

    # Load checkpoint if it exists
    it = utils.get_most_recent(checkpoint_dir, 'model') if args.model_it == -1 else args.model_it
    it, epoch_idx, loaded_clusterer = checkpoint_io.load_models(it=it, load_samples='supervised' != config['clusterer']['name'])

    if loaded_clusterer is None:
        print('Initializing new clusterer. The first clustering can be quite slow.')
        clusterer.recluster(discriminator=discriminator)
        checkpoint_io.save_clusterer(clusterer, it=0)
        np.savez(os.path.join(checkpoint_dir, 'cluster_samples.npz'), x=x_cluster)
    else:
        print('Using loaded clusterer')
        clusterer = loaded_clusterer

    # Evaluator
    evaluator = Evaluator(
        generator_test,
        zdist,
        ydist,
        train_loader=train_loader,
        clusterer=clusterer,
        batch_size=batch_size,
        device=device,
        inception_nsamples=config['training']['inception_nsamples'])

    # Trainer
    trainer = Trainer(generator,
                      discriminator,
                      g_optimizer,
                      d_optimizer,
                      gan_type=config['training']['gan_type'],
                      reg_type=config['training']['reg_type'],
                      reg_param=config['training']['reg_param'])

    # Training loop
    print('Start training...')
    while it < args.nepochs * len(train_loader):
        epoch_idx += 1

        for x_real, y in train_loader:
            it += 1

            x_real, y = x_real.to(device), y.to(device)
            z = zdist.sample((batch_size, ))
            y = clusterer.get_labels(x_real, y).to(device)

            # Discriminator updates
            dloss, reg = trainer.discriminator_trainstep(x_real, y, z)
            logger.add('losses', 'discriminator', dloss, it=it)
            logger.add('losses', 'regularizer', reg, it=it)

            # Generators updates
            gloss = trainer.generator_trainstep(y, z)
            logger.add('losses', 'generator', gloss, it=it)

            if config['training']['take_model_average']:
                update_average(generator_test, generator, beta=config['training']['model_average_beta'])

            # Print stats
            if it % log_every == 0:
                g_loss_last = logger.get_last('losses', 'generator')
                d_loss_last = logger.get_last('losses', 'discriminator')
                d_reg_last = logger.get_last('losses', 'regularizer')
                print('[epoch %0d, it %4d] g_loss = %.4f, d_loss = %.4f, reg=%.4f'
                      % (epoch_idx, it, g_loss_last, d_loss_last, d_reg_last))

            if it % config['training']['recluster_every'] == 0 and it > config['training']['burnin_time']:
                # print cluster distribution for online methods
                if it % 100 == 0 and config['training']['recluster_every'] <= 100:
                    print(f'[epoch {epoch_idx}, it {it}], distribution: {clusterer.get_label_distribution(x_real)}')
                clusterer.recluster(discriminator=discriminator, x_batch=x_real)

            # (i) Sample if necessary
            if it % config['training']['sample_every'] == 0:
                print('Creating samples...')
                x = evaluator.create_samples(z_test, y_test)
                x = evaluator.create_samples(z_test, clusterer.get_labels(x_test, y_test).to(device))
                logger.add_imgs(x, 'all', it)

                for y_inst in range(sample_nlabels):
                    x = evaluator.create_samples(z_test, y_inst)
                    logger.add_imgs(x, '%04d' % y_inst, it)

            # (ii) Compute inception if necessary
            if it % inception_every == 0 and it > 0:
                print('PyTorch Inception score...')
                inception_mean, inception_std = evaluator.compute_inception_score()
                logger.add('metrics', 'pt_inception_mean', inception_mean, it=it)
                logger.add('metrics', 'pt_inception_stddev', inception_std, it=it)
                print(f'[epoch {epoch_idx}, it {it}] pt_inception_mean: {inception_mean}, pt_inception_stddev: {inception_std}')

            # (iii) Backup if necessary
            if it % backup_every == 0:
                print('Saving backup...')
                checkpoint_io.save('model_%08d.pt' % it, it=it)
                checkpoint_io.save_clusterer(clusterer, int(it))
                logger.save_stats('stats_%08d.p' % it)

                if it > 0:
                    checkpoint_io.save('model.pt', it=it)


if __name__ == '__main__':
    exit_if_job_done(out_dir)
    main()
    mark_job_done(out_dir)
