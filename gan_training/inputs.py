import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

import os
import torch.utils.data as data
from torchvision.datasets.folder import default_loader
from PIL import Image
import random

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_dataset(name,
                data_dir,
                size=64,
                lsun_categories=None,
                deterministic=False,
                transform=None):
                
    transform = transforms.Compose([
        t for t in [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            (not deterministic) and transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            (not deterministic) and
            transforms.Lambda(lambda x: x + 1. / 128 * torch.rand(x.size())),
        ] if t is not False
    ]) if transform == None else transform

    if name == 'image':
        print('Using image labels')
        dataset = datasets.ImageFolder(data_dir, transform)
        nlabels = len(dataset.classes)
    elif name == 'webp':
        print('Using no labels from webp')
        dataset = CachedImageFolder(data_dir, transform)
        nlabels = len(dataset.classes)
    elif name == 'npy':
        # Only support normalization for now
        dataset = datasets.DatasetFolder(data_dir, npy_loader, ['npy'])
        nlabels = len(dataset.classes)
    elif name == 'cifar10':
        dataset = datasets.CIFAR10(root=data_dir,
                                   train=True,
                                   download=True,
                                   transform=transform)
        nlabels = 10
    elif name == 'stacked_mnist':
        dataset = StackedMNIST(data_dir,
                               transform=transforms.Compose([
                                   transforms.Resize(size),
                                   transforms.CenterCrop(size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, ), (0.5, ))
                               ]))
        nlabels = 1000
    elif name == 'lsun':
        if lsun_categories is None:
            lsun_categories = 'train'
        dataset = datasets.LSUN(data_dir, lsun_categories, transform)
        nlabels = len(dataset.classes)
    elif name == 'lsun_class':
        dataset = datasets.LSUNClass(data_dir,
                                     transform,
                                     target_transform=(lambda t: 0))
        nlabels = 1
    else:
        raise NotImplemented
    return dataset, nlabels

class CachedImageFolder(data.Dataset):
    """
    A version of torchvision.dataset.ImageFolder that takes advantage
    of cached filename lists.
    photo/park/004234.jpg
    photo/park/004236.jpg
    photo/park/004237.jpg
    """

    def __init__(self, root, transform=None, loader=default_loader):
        classes, class_to_idx = find_classes(root)
        self.imgs = make_class_dataset(root, class_to_idx)
        if len(self.imgs) == 0:
            raise RuntimeError("Found 0 images within: %s" % root)
        self.root = root
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, classidx = self.imgs[index]
        source = self.loader(path)
        if self.transform is not None:
            source = self.transform(source)
        return source, classidx

    def __len__(self):
        return len(self.imgs)

class StackedMNIST(data.Dataset):
    def __init__(self, data_dir, transform, batch_size=100000):
        super().__init__()
        self.channel1 = datasets.MNIST(data_dir,
                                       transform=transform,
                                       train=True,
                                       download=True)
        self.channel2 = datasets.MNIST(data_dir,
                                       transform=transform,
                                       train=True,
                                       download=True)
        self.channel3 = datasets.MNIST(data_dir,
                                       transform=transform,
                                       train=True,
                                       download=True)
        self.indices = {
            k: (random.randint(0,
                               len(self.channel1) - 1),
                random.randint(0,
                               len(self.channel1) - 1),
                random.randint(0,
                               len(self.channel1) - 1))
            for k in range(batch_size)
        }

    def __getitem__(self, index):
        index1, index2, index3 = self.indices[index]
        x1, y1 = self.channel1[index1]
        x2, y2 = self.channel2[index2]
        x3, y3 = self.channel3[index3]
        return torch.cat([x1, x2, x3], dim=0), y1 * 100 + y2 * 10 + y3

    def __len__(self):
        return len(self.indices)
        

def is_npy_file(path):
    return path.endswith('.npy') or path.endswith('.NPY')


def walk_image_files(rootdir):
    print(rootdir)
    if os.path.isfile('%s.txt' % rootdir):
        print('Loading file list from %s.txt instead of scanning dir' %
              rootdir)
        basedir = os.path.dirname(rootdir)
        with open('%s.txt' % rootdir) as f:
            result = sorted([
                os.path.join(basedir, line.strip()) for line in f.readlines()
            ])
            import random
            random.Random(1).shuffle(result)
            return result
    result = []

    IMG_EXTENSIONS = [
        '.jpg',
        '.JPG',
        '.jpeg',
        '.JPEG',
        '.png',
        '.PNG',
        '.ppm',
        '.PPM',
        '.bmp',
        '.BMP',
    ]

    for dirname, _, fnames in sorted(os.walk(rootdir)):
        for fname in sorted(fnames):
            if any(fname.endswith(extension)
                   for extension in IMG_EXTENSIONS) or is_npy_file(fname):
                result.append(os.path.join(dirname, fname))
    return result


def find_classes(dir):
    classes = [
        d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))
    ]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_class_dataset(source_root, class_to_idx):
    """
    Returns (source, classnum, feature)
    """
    imagepairs = []
    source_root = os.path.expanduser(source_root)
    for path in walk_image_files(source_root):
        classname = os.path.basename(os.path.dirname(path))
        imagepairs.append((path, 0))
    return imagepairs


def npy_loader(path):
    img = np.load(path)

    if img.dtype == np.uint8:
        img = img.astype(np.float32)
        img = img / 127.5 - 1.
    elif img.dtype == np.float32:
        img = img * 2 - 1.
    else:
        raise NotImplementedError

    img = torch.Tensor(img)
    if len(img.size()) == 4:
        img.squeeze_(0)

    return img
