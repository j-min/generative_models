import torch
import torch.nn as nn
import torch.nn.init
import torch.optim as optim
from torch.autograd import Variable

# Model
class VAE(nn.Module):
    def __init__(self, config):
        super(VAE, self).__init__()
        self.config = config

        # Encoder:  P(Z|X)
        self.encoder = nn.Sequential(
            nn.Linear(config.x_size, config.h_size), # X => h
            nn.LeakyReLU(config.LeakyReLU_slope),
            nn.Linear(config.h_size, config.z_size * 2)) # h => z (2 for mean/std)

        # Decoder: P(X|Z)
        self.decoder = nn.Sequential(
            nn.Linear(config.z_size, config.h_size), # z => h
            nn.ReLU(),
            nn.Linear(config.h_size, config.x_size), # h => x
            nn.Sigmoid())

    def reparameterize(self, mu, log_variance):
        """Sample z via reparameterization"""
        std = log_variance.mul(0.5).exp_()

        # Sampling from gaussian distribution
        if self.config.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)

        z = eps.mul(std).add_(mu)
        return z

    def forward(self, x):
        # Encode X => Z
        mu_log_variance = self.encoder(x.view(-1, self.config.x_size))
        mu, log_variance = torch.chunk(mu_log_variance, chunks=2, dim=1)
        z = self.reparameterize(mu, log_variance)

        # Reconstruct Z => X
        x_recon = self.decoder(z)
        return x_recon, mu, log_variance
