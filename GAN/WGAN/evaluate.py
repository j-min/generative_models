from solver import Solver
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
    config = get_config()
    main(config)
