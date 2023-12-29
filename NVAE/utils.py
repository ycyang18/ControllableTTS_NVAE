import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.nn.utils import spectral_norm

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)

class ChannelSELayer1d(nn.Module):
    def __init__(self, num_channels, reduction_ratio=4):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer1d, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.activ_1 = nn.ReLU()
        self.activ_2 = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H)
        :return: output tensor
        """
        batch_size, num_channels, H = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.activ_1(self.fc1(squeeze_tensor))
        fc_out_2 = self.activ_2(self.fc2(fc_out_1))

        return input_tensor * fc_out_2.view(batch_size, num_channels, 1)

class SELayer(nn.Module):
    def __init__(self, channel, reduction_ratio=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, l = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

# spectral normalization
def add_sn(m):
    '''
    Constraint/regulize Lipschitz constant (how much  model's weights change)
    -> the encoder output (latent codes) does not change dramatically as its input changes
    -> the latent codes predicted by the encoder remain bounded
    '''
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        return spectral_norm(m)
    else:
        return m

def reparameterize(mu, log_var):
    #prior = torch.distributions.Normal(0, 1) #
    #z = mu + prior.sample(mu.shape) * log_var
    z = mu + torch.randn_like(mu) * log_var # z = μ + (ε * σ) ##issue!!!!!!!!
    return z

def reparameterize_fix(mu, log_var):
    prior = torch.distributions.Normal(0, 1)
    z = mu + log_var * prior.sample(mu.shape).to(self.device)
    return z

#def reparameterize(mu, logvar):
#    std = torch.exp(0.5*logvar)
#    eps = torch.randn_like(std)
#    return mu + eps*std

def reparameterize_2(mu, log_var, prior):
    z = mu + log_var*prior.sample(mu.shape)
    return z

def combined_recon_loss(x, x_hat):
    loss = 0.1 * torch.nn.functional.l1_loss(x_hat, x) + \
            1.0 - torch.nn.functional.cosine_similarity(x_hat, x).mean() + \
            0.1 * torch.nn.functional.mse_loss(x_hat, x, reduction="mean")
    return loss

#def sigmoid_recon_loss(x, x_hat_sigmoid):
#    adaptive_loss = robust_loss_pytorch.adaptive.AdaptiveLossFunction(num_dims=1, float_dtype=np.float32, device="cpu")
#    recon_sig_loss = torch.mean(adaptive_loss.lossfun(
#            torch.mean(F.binary_cross_entropy(x_hat_sigmoid, x, reduction='none'), dim=[1, 2])[:, None]))

def kl(mu, log_var):
    """
    kl loss with standard norm distribute
    :param mu:
    :param log_var:
    :return:
    """
    loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim=[1,2]) #dim=[1,2,3]
    return torch.mean(loss, dim=0)

# kl_2_loss = kl_2(delta_mu, delta_log_var, mu, log_var)
def kl_2(delta_mu, delta_log_var, mu, log_var):
    var = torch.exp(log_var)
    delta_var = torch.exp(delta_log_var)
    loss = -0.5 * torch.sum(1 + delta_log_var - delta_mu ** 2 / var - delta_var, dim=[1, 2])
    return torch.mean(loss, dim=0)

def wasserstein_loss(mu, var):
    p = dist.Normal(torch.zeros_like(mu), torch.ones_like(var))
    q = dist.MultivariateNormal(mu, covariance_matrix=torch.diag_embed(var))
    w_distance = torch.sqrt((p.mean - q.mean).pow(2).sum() + torch.trace(p.covariance_matrix + q.covariance_matrix - 2*torch.sqrt(p.covariance_matrix @ q.covariance_matrix)))

    return w_distance


def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance 
    between two probability distributions
    """
    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2
    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2
    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance

'''
def kl_2(delta_mu, delta_log_var, mu, log_var):
    var = torch.exp(log_var)
    delta_var = torch.exp(delta_log_var)

    loss = -0.5 * torch.sum(1 + delta_var - delta_mu ** 2 / log_var - delta_log_var, dim=[1, 2]) #dim=[1,2,3]
    return torch.mean(loss, dim=0)
'''

class Annealer:
    def __init__(self, total_steps, shape, baseline=0.0, cyclical=True, disable=False):
        self.total_steps = total_steps
        self.current_step = 0
        self.cyclical = cyclical
        self.shape = shape
        self.baseline = baseline
        if disable:
            self.shape = 'none'
            self.baseline = 0.0

    def __call__(self, kl):
        out = kl * self.slope()
        return out

    def slope(self):
        if self.shape == 'linear':
            y = (self.current_step / self.total_steps)
        elif self.shape == 'cosine':
            y = (math.cos(math.pi * (self.current_step / self.total_steps - 1)) + 1) / 2
        elif self.shape == 'logistic':
            exponent = ((self.total_steps / 2) - self.current_step)
            y = 1 / (1 + math.exp(exponent))
        elif self.shape == 'none':
            y = 1.0
        else:
            raise ValueError('Invalid shape for annealing function. Must be linear, cosine, or logistic.')
        y = self.add_baseline(y)
        return y

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out

    def cyclical_setter(self, value):
        if value is not bool:
            raise ValueError('Cyclical_setter method requires boolean argument (True/False)')
        else:
            self.cyclical = value
        return