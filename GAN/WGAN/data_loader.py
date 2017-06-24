from torch.utils import data
from torchvision import transforms, datasets

def mnist_data_loader(image_path, batch_size, train=True, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.MNIST(image_path,
        train=train,
        download=False,
        transform=transform)

    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    return data_loader

def get_data_loader(image_path, image_size, batch_size, num_workers=2):
    transform = transforms.Compose([
        transforms.Scale(image_size),
        transforms.ToTensor(),
        # [0, 1] -> [-1, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder(
        root=image_path,
        transform=transform)

    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    return data_loader
