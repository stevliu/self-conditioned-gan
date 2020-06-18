from __future__ import absolute_import, division, print_function
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from scipy import linalg
import pathlib
import urllib
from tqdm import tqdm
import warnings


def check_or_download_inception(inception_path):
    ''' Checks if the path to the inception file is valid, or downloads
        the file if it is not present. '''
    INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    if inception_path is None:
        inception_path = '/tmp'
    inception_path = pathlib.Path(inception_path)
    model_file = inception_path / 'classify_image_graph_def.pb'
    if not model_file.exists():
        print("Downloading Inception model")
        from urllib import request
        import tarfile
        fn, _ = request.urlretrieve(INCEPTION_URL)
        with tarfile.open(fn, mode='r') as f:
            f.extract('classify_image_graph_def.pb', str(model_file.parent))
    return str(model_file)


def create_inception_graph(pth):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.io.gfile.GFile(pth, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='FID_Inception_Net')


def calculate_activation_statistics(images,
                                    sess,
                                    batch_size=200,
                                    verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations(images, sess, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


# code for handling inception net derived from
#   https://github.com/openai/improved-gan/blob/master/inception_score/model.py
def _get_inception_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    layername = 'FID_Inception_Net/pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims != []:
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    return pool3


#-------------------------------------------------------------------------------


def get_activations(images, sess, batch_size=200, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    n_images = images.shape[0]
    if batch_size > n_images:
        print(
            "warning: batch size is bigger than the data size. setting batch size to data size"
        )
        batch_size = n_images
    n_batches = n_images // batch_size
    pred_arr = np.empty((n_images, 2048))
    for i in tqdm(range(n_batches)):
        if verbose:
            print("\rPropagating batch %d/%d" % (i + 1, n_batches),
                  end="",
                  flush=True)
        start = i * batch_size

        if start + batch_size < n_images:
            end = start + batch_size
        else:
            end = n_images

        batch = images[start:end]
        pred = sess.run(inception_layer,
                        {'FID_Inception_Net/ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch_size, -1)
    if verbose:
        print(" done")
    return pred_arr


#-------------------------------------------------------------------------------


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(
        sigma2) - 2 * tr_covmean


def compute_fid_from_npz(path):
    print(path)
    with np.load(path) as data:
        fake_imgs = data['fake']

        name = None
        for name in ['imagenet', 'cifar', 'places']:
            if name in path: 
                real_imgs = name
                break
        print('Inferred name', name)
        if name is None:
            real_imgs = data['real']
            
        if fake_imgs.shape[0] < 1000: return 0

    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        m1, s1 = calculate_activation_statistics(fake_imgs, sess)
        if isinstance(real_imgs, str):
            print(f'using cached image stats for {real_imgs}')
            with np.load(precomputed_stats[real_imgs]) as data:
                m2, s2 = data['m'], data['s']
        else:
            print('computing real images stats from scratch')
            m2, s2 = calculate_activation_statistics(real_imgs, sess)

    return calculate_frechet_distance(m1, s1, m2, s2)

precomputed_stats = {
    'places':
    'output/places_gt_stats.npz',
    'imagenet':
    'output/imagenet_gt_stats.npz',
    'cifar':
    'output/cifar_gt_stats.npz'
}


def compute_fid_from_imgs(fake_imgs, real_imgs):
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        m1, s1 = calculate_activation_statistics(fake_imgs, sess)
        if isinstance(real_imgs, str):
            with np.load(precomputed_stats[real_imgs]) as data:
                m2, s2 = data['m'], data['s']
        else:
            m2, s2 = calculate_activation_statistics(real_imgs, sess)
    return calculate_frechet_distance(m1, s1, m2, s2)

def compute_stats(exp_path):
    #TODO: a bit hacky
    if 'places' in exp_path and not os.path.exists(precomputed_stats['places']):
        with np.load('output/places_gt_imgs.npz') as data_real:
            real_imgs = data_real['real']
            print('loaded real places images', real_imgs.shape)
        inception_path = check_or_download_inception(None)
        create_inception_graph(inception_path)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            m, s = calculate_activation_statistics(real_imgs, sess)
        np.savez(precomputed_stats['places'], m=m, s=s)
    
    if 'imagenet' in exp_path and not os.path.exists(precomputed_stats['imagenet']):
        with np.load('output/imagenet_gt_imgs.npz') as data_real:
            real_imgs = data_real['real']
            print('loaded real imagenet images', real_imgs.shape)
        inception_path = check_or_download_inception(None)
        create_inception_graph(inception_path)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            m, s = calculate_activation_statistics(real_imgs, sess)
        np.savez(precomputed_stats['imagenet'], m=m, s=s)

    if 'cifar' in exp_path and not os.path.exists(precomputed_stats['cifar']):
        with np.load('output/cifar_gt_imgs.npz') as data_real:
            real_imgs = data_real['real']
            print('loaded real cifar images', real_imgs.shape)
        inception_path = check_or_download_inception(None)
        create_inception_graph(inception_path)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            m, s = calculate_activation_statistics(real_imgs, sess)
        np.savez(precomputed_stats['cifar'], m=m, s=s)

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser('compute TF FID')
    parser.add_argument('--samples', help='path to samples')
    parser.add_argument('--it', type=str, help='path to samples')
    parser.add_argument('--results_dir', help='path to results_dir')
    args = parser.parse_args()
    
    it = args.it
    results_dir = args.results_dir

    compute_stats(args.samples)
    mean = compute_fid_from_npz(args.samples)
    print(f'FID: {mean}')
    
    if args.results_dir is not None:
        with open(os.path.join(args.results_dir, 'fid_results.json')) as f:
            fid_results = json.load(f)

        fid_results[it] = mean
        print(f'{results_dir} iteration {it} FID: {mean}')
        
        with open(os.path.join(args.results_dir, 'fid_results.json'), 'w') as f:
            f.write(json.dumps(fid_results))