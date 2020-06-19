# Diverse Image Generation via Self-Conditioned GANs

#### [Project](http://selfcondgan.csail.mit.edu/) |   [Paper](http://selfcondgan.csail.mit.edu/preprint.pdf)

**Diverse Image Generation via Self-Conditioned GANs** <br>
[Steven Liu](http://people.csail.mit.edu/stevenliu/),
[Tongzhou Wang](https://ssnl.github.io/),
[David Bau](http://people.csail.mit.edu/davidbau/home/),
[Jun-Yan Zhu](http://people.csail.mit.edu/junyanz/),
[Antonio Torralba](http://web.mit.edu/torralba/www/) <br>
MIT, Adobe Research<br>
in CVPR 2020.

![Teaser](images/teaser.png)

Our proposed self-conditioned GAN model learns to perform clustering and image synthesis simultaneously. The model training
requires no manual annotation of object classes. Here, we visualize several discovered clusters for both Places365 (top) and ImageNet
(bottom). For each cluster, we show both real images and the generated samples conditioned on the cluster index.

## Getting Started

### Installation
- Clone this repo:
```bash
git clone https://github.com/stevliu/self-conditioned-gan.git
cd self-conditioned-gan
```

- Install the dependencies
```bash
conda create --name selfcondgan python=3.6
conda activate selfcondgan
conda install --file requirements.txt
conda install -c conda-forge tensorboardx
```
### Training and Evaluation
- Train a model on CIFAR:
```bash
python train.py configs/cifar/selfcondgan.yaml
```

- Visualize samples and inferred clusters:
```bash
python visualize_clusters.py configs/cifar/selfcondgan.yaml --show_clusters
```
The samples and clusters will be saved to `output/cifar/selfcondgan/clusters`. If this directory lies on an Apache server, you can open the URL to `output/cifar/selfcondgan/clusters/+lightbox.html` in the browser and visualize all samples and clusters in one webpage.

- Evaluate the model's FID:
You will need to first gather a set of ground truth train set images to compute metrics against.
```bash
python utils/get_gt_imgs.py --cifar
python metrics.py configs/cifar/selfcondgan.yaml --fid --every -1
```
You can also evaluate with other metrics by appending additional flags, such as Inception Score (`--inception`), the number of covered modes + reverse-KL divergence (`--modes`), and cluster metrics (`--cluster_metrics`).

## Pretrained Models

You can load and evaluate pretrained models on ImageNet and Places. If you have access to ImageNet or Places directories, first fill in paths to your ImageNet and/or Places dataset directories in `configs/imagenet/default.yaml` and `configs/places/default.yaml` respectively. You can use the following config files with the evaluation scripts, and the code will automatically download the appropriate models.

```bash
configs/pretrained/imagenet/selfcondgan.yaml
configs/pretrained/places/selfcondgan.yaml

configs/pretrained/imagenet/conditional.yaml
configs/pretrained/places/conditional.yaml

configs/pretrained/imagenet/baseline.yaml
configs/pretrained/places/baseline.yaml
```

## Evaluation
### Visualizations

To visualize generated samples and inferred clusters, run
```bash
python visualize_clusters.py config-file
```
You can set the flag `--show_clusters` to also visualize the real inferred clusters, but this requires that you have a path to training set images.

### Metrics
To obtain generation metrics, fill in paths to your ImageNet or Places dataset directories in `utils/get_gt_imgs.py` and then run
```bash
python utils/get_gt_imgs.py --imagenet --places
```
to precompute batches of GT images for FID/FSD evaluation.

Then, you can use
```bash
python metrics.py config-file
```
with the appropriate flags compute the FID (`--fid`), FSD (`--fsd`), IS (`--inception`), number of modes covered/ reverse-KL divergence (`--modes`) and clustering metrics (`--cluster_metrics`) for each of the checkpoints.

## Training models
To train a model, set up a configuration file (examples in `/configs`), and run
```bash
python train.py config-file
```

An example config of self-conditioned GAN on ImageNet is `config/imagenet/selfcondgan.yaml` and on Places is `config/places/selfcondgan.yaml`.

Some models may be too large to fit on one GPU, so you may want to add `--devices DEVICE_NUMBERS` as an additional flag to do multi GPU training.

## 2D-experiments
For synthetic dataset experiments, first go into the `2d_mix` directory.

To train a self-conditioned GAN on the 2D-ring and 2D-grid dataset, run
```bash
python train.py --clusterer selfcondgan --data_type ring
python train.py --clusterer selfcondgan --data_type grid
```
You can test several other configurations via the command line arguments.


## Acknowledgments
This code is heavily based on the [GAN-stability](https://github.com/LMescheder/GAN_stability) code base.
Our FSD code is taken from the [GANseeing](https://github.com/davidbau/ganseeing) work.
To compute inception score, we use the code provided from [Shichang Tang](https://github.com/tsc2017/Inception-Score.git).
To compute FID, we use the code provided from [TTUR](https://github.com/bioinf-jku/TTUR).
We also use pretrained classifiers given by the [pytorch-playground](https://github.com/aaron-xichen/pytorch-playground).

We thank all the authors for their useful code.

## Citation
If you use this code for your research, please cite the following work.
```
@inproceedings{liu2020selfconditioned,
 title={Diverse Image Generation via Self-Conditioned GANs},
 author={Liu, Steven and Wang, Tongzhou and Bau, David and Zhu, Jun-Yan and Torralba, Antonio},
 booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
 year={2020}
}
```
