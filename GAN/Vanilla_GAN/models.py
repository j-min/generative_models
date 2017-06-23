import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.x_size, config.D_h1_size),
            nn.LeakyReLU(config.LeakyReLU_slope),
            nn.Linear(config.D_h1_size, config.D_h2_size),
            nn.LeakyReLU(config.LeakyReLU_slope),
            nn.Linear(config.D_h2_size, 1),
            nn.Sigmoid())

    def forward(self, x):
        return self.net(x)

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.z_size, config.G_h1_size),
            nn.LeakyReLU(config.LeakyReLU_slope),
            nn.Linear(config.G_h1_size, config.G_h2_size),
            nn.LeakyReLU(config.LeakyReLU_slope),
            nn.Linear(config.G_h2_size, config.x_size),
            nn.Tanh())

    def forward(self, x):
        return self.net(x)
