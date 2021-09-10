import params
from torch.utils import data
from torchvision import datasets, transforms
from cutstom_transforms.GeneralTransforms import WhiteNoise


def get_16_class_imageNet_dataloader(train=True, **kwarg):
    """
    **kwarg:
        gaussian_blur: bool, whether or not to use Gaussian_blur,
                        absence denote False
    """

    gaussian_blur = False
    target = False
    for key, value in kwarg.items():
        if key == "gaussian_blur":
            gaussian_blur = value
        elif key == "target":
            target = value

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if not gaussian_blur:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
                WhiteNoise(-0.5, 0.5),
            ]
        )

    if train:
        data_loader = data.DataLoader(
            dataset=datasets.ImageFolder(
                "data/16_class_source"
                if not target
                else "data/16_class_target",
                transform=transform,
            ),
            num_workers=2,
            batch_size=params.batch_size,
            shuffle=True,
            drop_last=True,
        )
    else:
        data_loader = data.DataLoader(
            dataset=datasets.ImageFolder(
                "data/16_class_source"
                if not target
                else "data/16_class_target",
                transform=transform,
            ),
            num_workers=2,
            batch_size=params.batch_size,
            shuffle=False,
            drop_last=False,
        )

    return data_loader
