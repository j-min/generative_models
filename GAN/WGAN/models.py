import torch.nn as nn

class MLP_D(nn.Module):
    def __init__(self, config):
        super(MLP_D, self).__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.x_size, config.D_h1_size),
            # nn.ReLU(),
            nn.LeakyReLU(config.LeakyReLU_slope),
            nn.Linear(config.D_h1_size, config.D_h2_size),
            # nn.ReLU(),
            nn.LeakyReLU(config.LeakyReLU_slope),
            nn.Linear(config.D_h2_size, 1)) # no sigmoid

    def forward(self, x):
        return self.net(x)

class MLP_G(nn.Module):
    def __init__(self, config):
        super(MLP_G, self).__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.z_size, config.G_h1_size),
            # nn.LeakyReLU(config.LeakyReLU_slope),
            nn.ReLU(),
            nn.Linear(config.G_h1_size, config.G_h2_size),
            # nn.LeakyReLU(config.LeakyReLU_slope),
            nn.ReLU(),
            nn.Linear(config.G_h2_size, config.x_size),
            # nn.Tanh(),
            )

    def forward(self, x):
        return self.net(x)

class DCGAN_D(nn.Module):
    def __init__(self, config):
        super(DCGAN_D, self).__init__()
        self.config = config
        self.net = nn.Sequential(
            # [n_channel, 28, 28] => [D_c_in, 14, 14]
            nn.Conv2d(in_channels=1, out_channels=D_c_in, kernel_size=5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # [D_c_in, 14, 14] => [D_c_in*2, 7, 7]
            nn.Conv2d(D_c_in, D_c_in*2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(D_c_in*2),
            nn.LeakyReLU(0.2, inplace=True),

            # [D_c_in*2, 7, 7] => [D_c_in*4, 4, 4]
            nn.Conv2d(D_c_in*2, D_c_in*4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(D_c_in*4),
            nn.LeakyReLU(0.2, inplace=True),

            # [D_c_in*4, 4, 4] => [1, 1]
            nn.Conv2d(D_c_in*4, 1, 4, 1, 0, bias=False),
        )


    def forward(self, x):
        return self.net(x).view(-1, 1)

class DCGAN_G(nn.Module):
    def __init__(self, config):
        super(DCGAN_G, self).__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.z_size, config.G_h1_size),
            # nn.BatchNorm1d(config.G_h1_size),
            nn.LeakyReLU(config.LeakyReLU_slope),
            nn.Linear(config.G_h1_size, config.G_h2_size),
            # nn.BatchNorm1d(config.G_h2_size),
            nn.LeakyReLU(config.LeakyReLU_slope),
            nn.Linear(config.G_h2_size, config.x_size),
            nn.Tanh())

    def forward(self, x):
        return self.net(x)
