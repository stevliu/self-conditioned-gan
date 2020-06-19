import argparse
import os
import json
from tqdm import tqdm

import numpy as np
import torch

from gan_training.config import load_config
from seeded_sampler import SeededSampler

parser = argparse.ArgumentParser('Computes numbers used in paper and caches them to a result files. Examples include FID, IS, reverse-KL, # modes, FSD, cluster NMI, Purity.')
parser.add_argument('paths', nargs='+', type=str, help='list of configs for each experiment')
parser.add_argument('--it', type=int, default=-1, help='If set, computes numbers only for that iteration')
parser.add_argument('--every', type=int, default=-1, help='skips some checkpoints and only computes those whose iteration number are divisible by every')
parser.add_argument('--fid', action='store_true', help='compute FID metric')
parser.add_argument('--inception', action='store_true', help='compute IS metric')
parser.add_argument('--modes', action='store_true', help='compute # modes and reverse-KL metric')
parser.add_argument('--fsd', action='store_true', help='compute FSD metric')
parser.add_argument('--cluster_metrics', action='store_true', help='compute clustering metrics (NMI, purity)')
parser.add_argument('--device', type=int, default=1, help='device to run the metrics on (can run into OOM issues if same as main device)')
args = parser.parse_args()

device = args.device
dirs = list(args.paths)

N = 50000
BS = 100

datasets = ['imagenet', 'cifar', 'stacked_mnist', 'places']

dataset_to_img = {
    'places': 'output/places_gt_imgs.npz',
    'imagenet': 'output/imagenet_gt_imgs.npz'}


def load_results(results_dir):
    results = []
    for results_file in ['fid_results.json', 'is_results.json', 'kl_results.json', 'nmodes_results.json', 'fsd_results.json', 'cluster_metrics.json']:
        results_file = os.path.join(results_dir, results_file)
        if not os.path.exists(results_file):
            with open(results_file, 'w') as f:
                f.write(json.dumps({}))
        with open(results_file) as f:
            results.append(json.load(f))
    return results


def get_dataset_from_path(path):
    for name in datasets:
        if name in path:
            print('Inferred dataset:', name)
            return name


def pt_to_np(imgs):
    '''normalizes pytorch image in [-1, 1] to [0, 255]'''
    return (imgs.permute(0, 2, 3, 1).mul_(0.5).add_(0.5).mul_(255)).clamp_(0, 255).numpy()


def sample(sampler):
    with torch.no_grad():
        samples = []
        for _ in tqdm(range(N // BS + 1)):
            x_real = sampler.sample(BS)[0].detach().cpu()
            x_real = [x.detach().cpu() for x in x_real]
            samples.extend(x_real)
        samples = torch.stack(samples[:N], dim=0)
        return pt_to_np(samples)


root = './'

while len(dirs) > 0:
    path = dirs.pop()
    if os.path.isdir(path):     # search down tree for config files
        for d1 in os.listdir(path):
            dirs.append(os.path.join(path, d1))
    else:
        if path.endswith('.yaml'):
            config = load_config(path, default_path='configs/default.yaml')
            outdir = config['training']['out_dir']

            if not os.path.exists(outdir) and config['pretrained'] == {}:
                print('Skipping', path, 'outdir', outdir)
                continue

            results_dir = os.path.join(outdir, 'results')
            checkpoint_dir = os.path.join(outdir, 'chkpts')
            os.makedirs(results_dir, exist_ok=True)

            fid_results, is_results, kl_results, nmodes_results, fsd_results, cluster_results = load_results(results_dir)

            checkpoint_files = os.listdir(checkpoint_dir) if os.path.exists(checkpoint_dir) else []
            if config['pretrained'] != {}:
                checkpoint_files = checkpoint_files + ['pretrained']

            for checkpoint in checkpoint_files:
                if (checkpoint.endswith('.pt') and checkpoint != 'model.pt') or checkpoint == 'pretrained':
                    print('Computing for', checkpoint)
                    if 'model' in checkpoint:
                        # infer iteration number from checkpoint file w/o loading it
                        if 'model_' in checkpoint:
                            it = int(checkpoint.split('model_')[1].split('.pt')[0])
                        else:
                            continue
                        if args.every != 0 and it % args.every != 0:
                            continue
                        # iteration 0 is often useless, skip it
                        if it == 0 or args.it != -1 and it != args.it:
                            continue
                    elif checkpoint == 'pretrained':
                        it = 'pretrained'
                    it = str(it)

                    clusterer_path = os.path.join(root, checkpoint_dir, f'clusterer{it}.pkl')
                    #  don't save samples for each iteration for disk space
                    samples_path = os.path.join(outdir, 'results', 'samples.npz')

                    targets = []
                    if args.inception:
                        targets = targets + [is_results]
                    if args.fid:
                        targets = targets + [fid_results]
                    if args.modes:
                        targets = targets + [kl_results, nmodes_results]
                    if args.fsd:
                        targets = targets + [fsd_results]

                    if all([it in result for result in targets]):
                        print('Already generated', it, path)
                    else:
                        sampler = SeededSampler(path,
                                                model_path=os.path.join(root, checkpoint_dir, checkpoint),
                                                clusterer_path=clusterer_path,
                                                pretrained=config['pretrained'])
                        samples = sample(sampler)
                        dataset_name = get_dataset_from_path(path)
                        np.savez(samples_path, fake=samples, real=dataset_name)

                    arguments = f'--samples {samples_path} --it {it} --results_dir {results_dir}'
                    if args.fid and it not in fid_results:
                        os.system(f'CUDA_VISIBLE_DEVICES={device} python gan_training/metrics/fid.py {arguments}')
                    if args.inception and it not in is_results:
                        os.system(f'CUDA_VISIBLE_DEVICES={device} python gan_training/metrics/tf_is/inception_score.py {arguments}')
                    if args.modes and (it not in kl_results or it not in nmodes_results):
                        os.system(f'CUDA_VISIBLE_DEVICES={device} python utils/get_empirical_distribution.py {arguments} --dataset {dataset_name}')
                    if args.cluster_metrics and it not in cluster_results:
                        os.system(f'CUDA_VISIBLE_DEVICES={device} python cluster_metrics.py {path} --model_it {it}')
                    if args.fsd and it not in fsd_results:
                        gt_path = dataset_to_img[dataset_name]
                        os.system(f'CUDA_VISIBLE_DEVICES={device} python -m seeing.fsd {gt_path} {samples_path} --it {it} --results_dir {results_dir}')
