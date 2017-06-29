import os
import pprint
import argparse
import torch
from torch import optim

base_dir = os.path.abspath(os.path.dirname(__file__))
datasets_path = '/Users/jmin/workspace/ML/datasets/'
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}

class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]
                setattr(self, key, value)

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str

def get_config():
    """Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class."""

    parser = argparse.ArgumentParser()

    #================ Train ==============#
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--D_n_iter', type=int, default=100)

    # Discriminator
    parser.add_argument('--D_learning_rate', type=float, default=5e-5)
    parser.add_argument('--D_weight_clamp', type=float, default=0.01)

    # Generator
    parser.add_argument('--G_learning_rate', type=float, default=5e-5)

    # Opitimizer
    parser.add_argument('--optimizer', type=str, default='RMSprop') # optim.Adam
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Input Size
    parser.add_argument('--x_size', type=int, default=28*28) # MNIST
    parser.add_argument('--n_channel', type=int, default=1)

    #=============== Model ===============#
    parser.add_argument('--DCGAN', action='store_true')
    parser.add_argument('--LeakyReLU_slope', type=float, default=0.2)

    # MLP
    parser.add_argument('--D_h1_size', type=int, default=128)
    parser.add_argument('--D_h2_size', type=int, default=128)
    parser.add_argument('--G_h1_size', type=int, default=128)
    parser.add_argument('--G_h2_size', type=int, default=128)

    # DCGAN
    parser.add_argument('--D_c_in', type=int, default=64)
    parser.add_argument('--G_c_in', type=int, default=64)


    # Latent z (noise)
    parser.add_argument('--z_size', type=int, default=64)

    #=============== Path ===============#
    parser.add_argument('--image_path', type=str, default=datasets_path)
    parser.add_argument('--image_log_path', type=str, default=os.path.join(base_dir, 'logs/'))
    parser.add_argument('--save_dir', type=str, default=os.path.join(base_dir, 'checkpoints/'))

    #=============== Misc ===============#
    parser.add_argument('--cuda', action='store_true', default=torch.cuda.is_available())
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=1)

    #=============== Parse Arguments===============#
    kwargs = parser.parse_args()

    # Namespace => Dictionary
    kwargs = vars(kwargs)

    return Config(**kwargs)
