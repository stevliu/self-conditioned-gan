''' Samples from a (class-conditional) GAN, so that the samples can be reproduced '''

import os
import pickle
import random
import copy

import torch
from torch import nn

from gan_training.checkpoints import CheckpointIO
from gan_training.config import (load_config, build_models)
from seeing.yz_dataset import YZDataset


def get_most_recent(models):
    model_numbers = [
        int(model.split("model.pt")[0]) if model != "model.pt" else 0
        for model in models
    ]
    return str(max(model_numbers)) + "model.pt"


class SeededSampler():
    def __init__(
            self,
            config_name,        # name of experiment's config file
            model_path="",      # path to the model. empty string infers the most recent checkpoint
            clusterer_path="",  # path to the clusterer, ignored if gan type doesn't require a clusterer
            pretrained={},      # urls to the pretrained models
            rootdir='./',
            device='cuda:0'):
        self.config = load_config(os.path.join(rootdir, config_name), 'configs/default.yaml')
        self.model_path = model_path
        self.clusterer_path = clusterer_path
        self.rootdir = rootdir
        self.nlabels = self.config['generator']['nlabels']
        self.device = device
        self.pretrained = pretrained

        self.generator = self.get_generator()
        self.generator.eval()
        self.yz_dist = self.get_yz_dist()

    def sample(self, nimgs):
        '''
        samples an image using the generator, with z drawn from isotropic gaussian, and y drawn from self.yz_dist.
        For baseline methods, y doesn't matter because y is ignored in the input
        yz_dist is the empirical label distribution for the clustered gans.

        returns the image, and the integer seed used to generate it. generated sample is in [-1, 1]
        '''
        self.generator.eval()
        with torch.no_grad():
            seeds = [random.randint(0, 1e8) for _ in range(nimgs)]
            z, y = self.yz_dist(seeds)
            return self.generator(z, y), seeds

    def conditional_sample(self, yi, seed=None):
        ''' returns a generated sample, which is in [-1, 1], seed is an int'''
        self.generator.eval()
        with torch.no_grad():
            if seed is None:
                seed = [random.randint(0, 1e8)]
            else:
                seed = [seed]
            z, _ = self.yz_dist(seed)
            y = torch.LongTensor([yi]).to(self.device)
            return self.generator(z, y)

    def sample_with_seed(self, seeds):
        ''' returns a generated sample, which is in [-1, 1] '''
        self.generator.eval()
        z, y = self.yz_dist(seeds)
        return self.generator(z, y)

    def get_zy(self, seeds):
        '''returns the batch of z, y corresponding to the seeds'''
        return self.yz_dist(seeds)

    def sample_with_zy(self, z, y):
        ''' returns a generated sample given z and y, which is in [-1, 1].'''
        self.generator.eval()
        return self.generator(z, y)

    def get_generator(self):
        ''' loads a generator according to self.model_path '''

        exp_out_dir = os.path.join(self.rootdir, self.config['training']['out_dir'])
        # infer checkpoint if neeeded
        checkpoint_dir = os.path.join(exp_out_dir, 'chkpts') if self.model_path == "" or 'model' in self.pretrained else "./"
        model_name = get_most_recent(os.listdir(checkpoint_dir)) if self.model_path == "" else self.model_path

        checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)
        self.checkpoint_io = checkpoint_io

        generator, _ = build_models(self.config)
        generator = generator.to(self.device)
        generator = nn.DataParallel(generator)

        if self.config['training']['take_model_average']:
            generator_test = copy.deepcopy(generator)
            checkpoint_io.register_modules(generator_test=generator_test)
        else:
            generator_test = generator

        checkpoint_io.register_modules(generator=generator)

        try:
            it = checkpoint_io.load(model_name, pretrained=self.pretrained)
            assert (it != -1)
        except Exception as e:
            # try again without data parallel
            print(e)
            checkpoint_io.register_modules(generator=generator.module)
            checkpoint_io.register_modules(generator_test=generator_test.module)
            it = checkpoint_io.load(model_name, pretrained=self.pretrained)
            assert (it != -1)

        print('Loaded iteration:', it['it'])
        return generator_test

    def get_yz_dist(self):
        '''loads the z and y dists used to sample from the generator.'''

        if self.config['clusterer']['name'] != 'supervised':
            if 'clusterer' in self.pretrained:
                clusterer = self.checkpoint_io.load_clusterer('pretrained', load_samples=False, pretrained=self.pretrained)
            elif os.path.exists(self.clusterer_path):
                with open(self.clusterer_path, 'rb') as f:
                    clusterer = pickle.load(f)

            if isinstance(clusterer.discriminator, nn.DataParallel):
                clusterer.discriminator = clusterer.discriminator.module

            if clusterer.kmeans is not None:
                # use clusterer empirical distribution as sampling
                print('Using k-means empirical distribution')
                distribution = clusterer.get_label_distribution()
                probs = [f / sum(distribution) for f in distribution]
            else:
                # otherwise, use a uniform distribution. this is not desired, unless it's a random label or unconditional GAN
                print("Sampling with uniform distribution over", clusterer.k, "labels")
                probs = [1. / clusterer.k for _ in range(clusterer.k)]
        else:
            # if it's supervised, then sample uniformly over all classes.
            # this might not be the right thing to do, since datasets are usually imbalanced.
            print("Sampling with uniform distribution over", self.nlabels,
                  "labels")
            probs = [1. / self.nlabels for _ in range(self.nlabels)]

        return YZDataset(zdim=self.config['z_dist']['dim'],
                         nlabels=len(probs),
                         distribution=probs,
                         device=self.device)
