import random
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .image_folder import make_dataset
from .image import base_transform, get_transform


def get_dataset(config):
    """Return dataset class"""

    torchvision_datasets = [
        'LSUN',
        'CocoCaptions',
        'CocoDetection',
        'CIFAR10',
        'CIFAR100',
        'FashionMNIST',
        'MNIST',
        'STL10',
        'SVHN',
        'PhotoTour',
        'SEMEION']

    # unaligned_datasets = [
    #     'horse2zebra'
    # ]

    if config.dataset in torchvision_datasets:
        dataset = getattr(datasets, config.dataset)(
            root=config.dataset_dir,
            train=config.isTrain,
            download=True,
            transform=base_transform(config))
    else:
        dataset = get_custom_dataset(config)
    return dataset


def get_custom_dataset(config):
    dataset = None
    if config.dataset_mode == 'aligned':
        dataset = AlignedDataset()
    elif config.dataset_mode == 'unaligned':
        dataset = UnalignedDataset()
    elif config.dataset_mode == 'single':
        dataset = SingleDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % config.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(config)
    return dataset


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, config):
        pass


class UnalignedDataset(BaseDataset):
    def initialize(self, config):
        self.config = config
        self.root_dir = config.dataset_dir
        self.dir_A = self.root_dir.joinpath(config.mode + 'A')
        self.dir_B = self.root_dir.joinpath(config.mode + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(
            config.resize_crop,
            config.flip,
            config.loadSize,
            config.fineSize)

    def __getitem__(self, index):
        index_A = index % self.A_size
        A_path = self.A_paths[index_A]
        if self.config.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform(A_img)
        B = self.transform(B_img)
        if self.config.which_direction == 'BtoA':
            input_nc = self.config.output_nc
            output_nc = self.config.input_nc
        else:
            input_nc = self.config.input_nc
            output_nc = self.config.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'


class AlignedDataset(BaseDataset):
    def initialize(self, config):
        self.config = config
        self.root_dir = config.dataset_dir
        self.dir_AB = self.root_dir.joinpath(config.mode)

        self.AB_paths = sorted(make_dataset(self.dir_AB))

        assert(config.resize_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.config.loadSize * 2, self.config.loadSize), Image.BICUBIC)
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.config.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.config.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.config.fineSize,
               w_offset:w_offset + self.config.fineSize]
        B = AB[:, h_offset:h_offset + self.config.fineSize,
               w + w_offset:w + w_offset + self.config.fineSize]

        if self.config.which_direction == 'BtoA':
            input_nc = self.config.output_nc
            output_nc = self.config.input_nc
        else:
            input_nc = self.config.input_nc
            output_nc = self.config.output_nc

        if (not self.config.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'


class SingleDataset(BaseDataset):
    def initialize(self, config):
        self.config = config
        self.root_dir = config.dataset_dir
        self.dir_A = self.root_dir

        self.A_paths = make_dataset(self.dir_A)

        self.A_paths = sorted(self.A_paths)

        self.transform = get_transform(config)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)
        if self.config.which_direction == 'BtoA':
            input_nc = self.config.output_nc
        else:
            input_nc = self.config.input_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'
