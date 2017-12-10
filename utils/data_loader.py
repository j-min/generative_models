from torch.utils.data import DataLoader
from .datasets import get_dataset, get_custom_dataset


class BaseDataLoader():
    def __init__(self):
        pass

    def initialize(self, config):
        self.config = config
        pass

    def load_data():
        return None


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, config):
        BaseDataLoader.initialize(self, config)
        self.dataset = get_custom_dataset(config)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batchSize,
            shuffle=not config.serial_batches,
            num_workers=int(config.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.config.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.config.max_dataset_size:
                break
            yield data


def get_data_loader(config):
    dataset = get_dataset(config)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=config.isTrain,
        num_workers=config.nThreads)

    return data_loader
