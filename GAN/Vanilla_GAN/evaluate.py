import argparse
from solver import Solver
# from torch.backends import cudnn
from data_loader import mnist_data_loader
from configs import get_config

def main(config):
    data_loader = mnist_data_loader(
        image_path=config.image_path,
        batch_size=config.batch_size,
        train=False)

    solver = Solver(config, data_loader=data_loader, is_train=False)
    print(config)
    solver.build()
    solver.eval()

if __name__ == '__main__':
    # Get Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', action='store', type=int, default=100)
    parser.add_argument('--epochs', action='store', type=int, default=20)
    kwargs = parser.parse_args()

    # Namespace => dictionary
    kwargs = vars(kwargs)

    config = get_config(**kwargs)

    main(config)
