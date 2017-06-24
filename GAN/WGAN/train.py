import argparse
import os
from solver import Solver
# from torch.backends import cudnn
from data_loader import mnist_data_loader
from configs import get_config

def main(config):
    data_loader = mnist_data_loader(
        image_path=config.image_path,
        batch_size=config.batch_size,
        train=True)

    solver = Solver(config, data_loader=data_loader, is_train=True)
    print(config)
    print(f'\nTotal data size: {solver.total_data_size}\n')

    solver.build()
    solver.train()

if __name__ == '__main__':
    # Get Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', action='store', type=int, default=100)
    parser.add_argument('--epochs', action='store', type=int, default=20)
    # TODO: add arguments
    kwargs = parser.parse_args()

    # Namespace => dictionary
    kwargs = vars(kwargs)

    config = get_config(**kwargs)


    for path in [config.image_path, config.image_log_path, config.save_dir]:
        if not os.path.isdir(path):
            os.mkdir(path)

    main(config)
