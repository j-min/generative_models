import torch
import torch.nn as nn
# from utils import to_var
from torch.autograd import Variable
import layers


class VAE(nn.Module):
    def __init__(self, config):
        """Vanilla VAE model"""
        super(VAE, self).__init__()

        self.config = config

        if config.layer == 'mlp':
            # Encoder:  q(Z|X)
            self.encoder = layers.MLP(dims=[config.x_size, config.h_size])
            self.relu = nn.ReLU()
            self.to_mu = nn.Linear(config.h_size, config.z_size)
            self.to_sigma = nn.Linear(config.h_size, config.z_size)

            # Decoder: q(X|Z)
            self.decoder = layers.MLP(dims=[config.z_size, 200, config.x_size])
            self.sigmoid = nn.Sigmoid()

        elif config.layer == 'conv':
            # Encoder:  q(Z|X)
            self.encoder = layers.ConvEncoder()
            self.relu = nn.ReLU()
            self.to_mu = nn.Linear(32 * 5 * 5, config.z_size)
            self.to_sigma = nn.Linear(32 * 5 * 5, config.z_size)

            # Decoder: q(X|Z)
            self.fc1 = layers.MLP(dims=[config.z_size, 32 * 5 * 5, 1568])
            self.decoder = layers.ConvDecoder()
            self.sigmoid = nn.Sigmoid()

    def sample(self, mu, log_variance):
        """Sample z via reparameterization"""
        sigma = torch.exp(0.5 * log_variance)
        # eps = to_var(torch.randn(sigma.size()))
        eps = Variable(torch.randn(sigma.size()))
        z = mu + sigma * eps
        return z

    def forward(self, x):
        """X => Z => X"""

        batch_size, n_channel, height, width = x.size()
        if self.config.layer == 'mlp':
            x = x.view(batch_size, -1)

        # Encode X => Z
        h = self.relu(self.encoder(x))
        if self.config.layer == 'conv':
            batch_size, n_channel, height, width = h.size()
            h = h.view(batch_size, -1)
        mu, log_variance = self.to_mu(h), self.to_sigma(h)

        # Sample Z
        z = self.sample(mu, log_variance)

        # Reconstruct Z => X
        if self.config.layer == 'conv':
            z = self.fc1(z)
            z = z.view(batch_size, 32, 7, 7)
        x_recon = self.sigmoid(self.decoder(z))

        return x_recon, mu, log_variance
