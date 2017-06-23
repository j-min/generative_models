import os
import pprint
import torch
from torch import optim

base_dir = os.path.abspath(os.path.dirname(__file__))
datasets_path = '/Users/jmin/workspace/ML/datasets/'

class TrainConfig(object):
    def __init__(self):
        self.batch_size = 100
        self.epochs = 10

        # Discriminator
        self.d_learning_rate = 1e-4#3*1e-5

        # Generator
        self.g_learning_rate = 1e-4

        self.optimizer = optim.Adam
        self.beta1 = 0.9
        self.beta2 = 0.999

        self.x_size = 28*28 # MNIST

class ModelConfig(object):
    def __init__(self):

        self.LeakyReLU_slope = 0.1

        # Discriminator
        self.D_h1_size = 256
        self.D_h2_size = 256

        # Generator
        self.G_h1_size = 256
        self.G_h2_size = 256

        # Noise
        self.z_size = 64

class PathConfig(object):
    def __init__(self):
        self.image_path = datasets_path
        self.image_log_path = os.path.join(base_dir, 'logs/')
        self.save_dir = os.path.join(base_dir, 'checkpoints/')

class MiscConfig(object):
    def __init__(self):
        self.cuda = torch.cuda.is_available()
        self.seed = 1
        self.log_interval = 100
        self.save_interval = 1

class Config(object):
    def __init__(self, **kwargs):
        TrainConfig.__init__(self)
        ModelConfig.__init__(self)
        PathConfig.__init__(self)
        MiscConfig.__init__(self)

        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def __str__(self):
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str

def get_config(**kwargs):
    return Config(**kwargs)
