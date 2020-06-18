import numpy as np
import torch
from torch.nn import functional as F

from gan_training.metrics import inception_score

class Evaluator(object):
    def __init__(self,
                 generator,
                 zdist,
                 ydist,
                 train_loader,
                 clusterer,
                 batch_size=64,
                 inception_nsamples=10000,
                 device=None):
        self.generator = generator
        self.clusterer = clusterer
        self.train_loader = train_loader
        self.zdist = zdist
        self.ydist = ydist
        self.inception_nsamples = inception_nsamples
        self.batch_size = batch_size
        self.device = device

    def sample_z(self, batch_size):
        return self.zdist.sample((batch_size, )).to(self.device)

    def get_y(self, x, y):
        return self.clusterer.get_labels(x, y).to(self.device)

    def get_fake_real_samples(self, N):
        ''' returns N fake images and N real images in pytorch form'''
        with torch.no_grad():
            self.generator.eval()
            fake_imgs = []
            real_imgs = []
            while len(fake_imgs) < N:
                for x_real, y_gt in self.train_loader:
                    x_real = x_real.cuda()
                    z = self.sample_z(x_real.size(0))
                    y = self.get_y(x_real, y_gt)
                    samples = self.generator(z, y)
                    samples = [s.data.cpu() for s in samples]
                    fake_imgs.extend(samples)
                    real_batch = [img.data.cpu() for img in x_real]
                    real_imgs.extend(real_batch)
                    assert (len(real_imgs) == len(fake_imgs))
                    if len(fake_imgs) >= N:
                        fake_imgs = fake_imgs[:N]
                        real_imgs = real_imgs[:N]
                        return fake_imgs, real_imgs

    def compute_inception_score(self):
        imgs, _ = self.get_fake_real_samples(self.inception_nsamples)
        imgs = [img.numpy() for img in imgs]
        score, score_std = inception_score(imgs,
                                           device=self.device,
                                           resize=True,
                                           splits=1)

        return score, score_std

    def create_samples(self, z, y=None):
        self.generator.eval()
        batch_size = z.size(0)
        # Parse y
        if y is None:
            raise NotImplementedError()
        elif isinstance(y, int):
            y = torch.full((batch_size, ),
                           y,
                           device=self.device,
                           dtype=torch.int64)
        # Sample x
        with torch.no_grad():
            x = self.generator(z, y)
        return x


