from .mnist import get_mnist
from .usps import get_usps
from .sixteen_class_imagenet import get_16_class_imageNet_dataloader

__all__ = (get_usps, get_mnist, get_16_class_imageNet_dataloader)
