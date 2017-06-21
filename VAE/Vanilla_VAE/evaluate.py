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
    solver.build_model()
    solver.eval()

if __name__ == '__main__':
    # Get Configuration
    parser = argparse.ArgumentParser()
    # TODO: add arguments
    kwargs = parser.parse_args()

    # Namespace => dictionary
    kwargs = vars(kwargs)

    config = get_config(**kwargs)
    config = get_config()
    print(config)
    main(config)
