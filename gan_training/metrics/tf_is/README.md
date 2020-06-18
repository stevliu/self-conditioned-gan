Inception Score
=====================================

A new Tensorflow implementation of the "Inception Score" (IS) for the evaluation of generative models, with a bug raised in [https://github.com/openai/improved-gan/issues/29](https://github.com/openai/improved-gan/issues/29) fixed. 

## Major Dependency
- `tensorflow >= 1.14`

## Features
- Fast, easy-to-use and memory-efficient, written in a way that is similar to the original implementation
- No prior knowledge about Tensorflow is necessary if your are using CPU or GPU
- Makes use of [TF-GAN](https://github.com/tensorflow/gan)
- Downloads InceptionV1 automatically
- Compatible with both Python 2 and Python 3

## Usage
- If you are working with GPU, use `inception_score.py`; if you are working with TPU, use `inception_score_tpu.py` and pass a Tensorflow Session and a [TPUStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/TPUStrategy) as additional arguments.
- Call `get_inception_score(images, splits=10)`, where `images` is a numpy array with values ranging from 0 to 255 and shape in the form `[N, 3, HEIGHT, WIDTH]` where `N`, `HEIGHT` and `WIDTH` can be arbitrary. `dtype` of the images is recommended to be `np.uint8` to save CPU memory.
- A smaller `BATCH_SIZE` reduces GPU/TPU memory usage, but at the cost of a slight slowdown.
- If you want to compute a general "Classifier Score" with probabilities `preds` from another classifier, call `preds2score(preds, splits=10)`. `preds` can be a numpy array of arbitrary shape `[N, num_classes]`.
## Links
- The Inception Score was proposed in the paper [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
- Code for the [Fréchet Inception Distance](https://github.com/tsc2017/Frechet-Inception-Distance)
