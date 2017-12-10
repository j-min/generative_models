from configs import Config
from solver import Solver
from utils.data_loader import get_data_loader


def main(config):
    train_loader = get_data_loader(config)

    solver = Solver(config, train_loader=train_loader, is_train=True)
    print(config)
    print(f'\nTotal data size: {solver.total_data_size}\n')

    solver.build()
    solver.train()


if __name__ == '__main__':
    # Get Configuration
    config = Config().initialize()
    # import ipdb
    # ipdb.set_trace()
    main(config)
