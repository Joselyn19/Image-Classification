import torch
import params

import torch.nn as nn

from torch import Tensor

from typing import Type, Any, Union, List

from torchvision.models.resnet import (
    ResNet,
    BasicBlock,
    Bottleneck,
    load_state_dict_from_url,
    model_urls,
)


class ResNet50Classifier(nn.Module):
    def __init__(self):

        super(ResNet50Classifier, self).__init__()

        self.fc = nn.Linear(2048, params.num_classes)
        self.restored = False
        self.fc.weight.data.normal_(0, 0.005)

    def forward(self, feat):

        return self.fc(feat)


class _ResNet50Encoder(ResNet):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        del self.fc
        self.restored = True

    def _forward_impl(self, x: Tensor) -> Tensor:

        # See note [TorchScript super()]

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        return x


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:

    model = _ResNet50Encoder(block, layers, **kwargs)
    if pretrained:
        print("using pretrained resnet50's encoder")
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress
        )

        model.load_state_dict(state_dict, strict=False)

    return model


def ResNet50Encoder(
    pretrained: bool = True, progress: bool = True, **kwargs: Any
) -> ResNet:

    r"""ResNet-50 model from


    `"Deep Residual Learning for Image Recognition"


        <https://arxiv.org/pdf/1512.03385.pdf>`_.



    Args:


        pretrained (bool): If True, returns a model pre-trained on ImageNet


        progress (bool): If True, displays a progress bar of the download
        to stderr


    """

    return _resnet(
        "resnet50", Bottleneck, [3, 4, 6, 3], True, progress, **kwargs
    )
