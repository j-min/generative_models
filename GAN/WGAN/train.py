import os
from solver import Solver
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
    solver.plot()
    solver.gif()

if __name__ == '__main__':
    config = get_config()
    for path in [config.image_path, config.image_log_path, config.save_dir]:
        if not os.path.isdir(path):
            os.mkdir(path)
    main(config)
