import torch


def np_to_pt(x):
    ''' permutes the appropriate channels to turn numpy formatted images to pt formatted images. does NOT renormalize '''
    x = torch.from_numpy(x)
    if len(x.shape) == 4:
        return x.permute(0, 3, 1, 2)
    elif len(x.shape) == 3:
        return x.permute(2, 0, 1)
    else:
        raise NotImplementedError
