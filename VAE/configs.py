import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from utils.configs import BaseConfig


class Config(BaseConfig):
    def parse(self):
        self.parser.add_argument('--h_size', type=int, default=200)
        self.parser.add_argument('--z_size', type=int, default=20)
        self.parser.add_argument('--recon_loss', type=str, default='l2')
        self.parser.add_argument('--layer', type=str, default='conv')


def get_config(**kwargs):
    return BaseConfig().initialize(**kwargs)
