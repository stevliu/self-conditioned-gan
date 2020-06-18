import torch

from models import generator_dict, discriminator_dict
from torch import optim
import torch.utils.data as utils


def get_models(model_type, conditioning, k_value, d_act_dim, device):
    G = generator_dict[model_type]
    D = discriminator_dict[model_type]
    generator = G(conditioning, k_value=k_value)
    discriminator = D(conditioning, k_value=k_value, act_dim=d_act_dim)

    generator.to(device)
    discriminator.to(device)

    return generator, discriminator


def get_optimizers(generator, discriminator, lr=1e-4, beta1=0.8, beta2=0.999):
    g_optimizer = optim.Adam(generator.parameters(),
                             lr=lr,
                             betas=(beta1, beta2))
    d_optimizer = optim.Adam(discriminator.parameters(),
                             lr=lr,
                             betas=(beta1, beta2))
    return g_optimizer, d_optimizer


def get_test(get_data, batch_size, variance, k_value, device):
    x_test, y_test = get_data(batch_size, var=variance)
    x_test, y_test = torch.from_numpy(x_test).float().to(
        device), torch.from_numpy(y_test).long().to(device)
    return x_test, y_test


def get_dataset(get_data, batch_size, npts, variance, k_value):
    samples, labels = get_data(npts, var=variance)
    tensor_samples = torch.stack([torch.Tensor(x) for x in samples])
    tensor_labels = torch.stack([torch.tensor(x) for x in labels])
    dataset = utils.TensorDataset(tensor_samples, tensor_labels)
    train_loader = utils.DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=True,
                                    sampler=None,
                                    drop_last=True)
    return train_loader
