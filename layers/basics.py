import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dims=[784, 200, 100]):
        """MLP with LeakyReLU"""
        super(MLP, self).__init__()
        self.dims = dims
        _layers = []
        _layers.append(nn.Linear(dims[0], dims[1]))
        for i in range(1, len(dims) - 1):
            _layers.append(nn.LeakyReLU(0.2, True))
            _layers.append(nn.Linear(dims[i], dims[i + 1]))

        self.net = nn.Sequential(*_layers)

    def forward(self, x):
        return self.net(x)


def convblock(in_channel, out_channel, ksize=3, stride=1, padding=1,
              batch_norm=True, activation='relu'):
    _layers = [
        nn.Conv2d(in_channel, out_channel, ksize, stride, padding)
    ]
    if batch_norm:
        _layers.append(nn.BatchNorm2d(out_channel))
    if activation.lower() == 'relu':
        _layers.append(nn.ReLU(True))
    elif activation.lower() == 'leakyrelu':
        _layers.append(nn.LeakyReLU(0.2, True))
    return nn.Sequential(*_layers)


class ConvEncoder(nn.Module):
    def __init__(self, dims=[1, 8, 16, 32], ksize=3):
        super(ConvEncoder, self).__init__()
        self.net = nn.Sequential(
            convblock(dims[0], dims[1]),
            nn.MaxPool2d(2, 2),

            convblock(dims[1], dims[2]),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(dims[2], dims[3], ksize),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def deconvblock(in_channel, out_channel, ksize=3, stride=2, padding=1, dilation=1,
                batch_norm=True, activation='relu'):
    _layers = [
        nn.ConvTranspose2d(in_channel, out_channel, ksize, stride, padding, dilation)
    ]

    if activation.lower() == 'relu':
        _layers.append(nn.ReLU(True))
    elif activation.lower() == 'leakyrelu':
        _layers.append(nn.LeakyReLU(0.2, True))
    if batch_norm:
        _layers.append(nn.BatchNorm2d(out_channel))
    return nn.Sequential(*_layers)


class ConvDecoder(nn.Module):
    def __init__(self, dims=[32, 16, 8, 1], ksize=3):
        super(ConvDecoder, self).__init__()
        self.net = nn.Sequential(
            deconvblock(dims[0], dims[1]),
            deconvblock(dims[1], dims[2]),
            nn.ConvTranspose2d(dims[2], dims[3], ksize, stride=1, padding=1),
            nn.BatchNorm2d(dims[3]),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
