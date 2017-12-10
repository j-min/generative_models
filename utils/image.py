from torchvision.utils import save_image as _save_image
from torchvision import transforms
from PIL import Image


def load_image(image_path):
    return Image.open(image_path)


def save_image(tensor, image_path, **kwargs):
    _save_image(tensor, image_path, **kwargs)


def denorm(x):
    """(-1, 1) => (0, 1)"""
    out = (x + 1) / 2
    return out.clamp(0, 1)


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)


def base_transform(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # [0, 1] -> [-1. 1]
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform


def get_transform(resize_crop='resize_and_crop', flip=True,
                  loadSize=286, fineSize=256):
    transform_list = []
    if resize_crop == 'resize_and_crop':
        osize = [loadSize, loadSize]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(fineSize))
    elif resize_crop == 'crop':
        transform_list.append(transforms.RandomCrop(fineSize))
    elif resize_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, fineSize)))
    elif resize_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, loadSize)))
        transform_list.append(transforms.RandomCrop(fineSize))

    if flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)
