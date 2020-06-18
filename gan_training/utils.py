import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision

import os


def save_images(imgs, outfile, nrow=8):
    imgs = imgs / 2 + 0.5  # unnormalize
    torchvision.utils.save_image(imgs, outfile, nrow=nrow)


def get_nsamples(data_loader, N):
    x = []
    y = []
    n = 0
    for x_next, y_next in data_loader:
        x.append(x_next)
        y.append(y_next)
        n += x_next.size(0)
        if n > N:
            break
    x = torch.cat(x, dim=0)[:N]
    y = torch.cat(y, dim=0)[:N]
    return x, y


def update_average(model_tgt, model_src, beta):
    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)


def get_most_recent(d, ext):
    if not os.path.exists(d):
        print('Directory', d, 'does not exist')
        return -1 
    its = []
    for f in os.listdir(d):
        try:
            it = int(f.split(ext + "_")[1].split('.pt')[0])
            its.append(it)
        except Exception as e:
            pass
    if len(its) == 0:
        print('Found no files with extension \"%s\" under %s' % (ext, d))
        return -1
    return max(its)
