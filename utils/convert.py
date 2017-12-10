import torch
from torch.autograd import Variable
import numpy as np


def to_var(x, eval=False, on_cpu=False, gpu_id=None, async=False):
    """Tensor => Variable"""
    if torch.cuda.is_available() and not on_cpu:
        x = x.cuda(gpu_id, async)
    if eval:
        x = Variable(x, volatile=True)
    else:
        x = Variable(x)
    return x


def to_tensor(x):
    """Variable => Tensor"""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data


def to_image(image_tensor, imtype=np.uint8, cvt_rgb=True):
    """Tensor => Numpy Array"""
    # Batch => First batch
    if len(image_tensor.size()) == 4:
        image_tensor = image_tensor[0]

    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1 and cvt_rgb:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)
