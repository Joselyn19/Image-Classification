# ADDA_CORAL
ADDA+CORAL in pretrain

## Configuration

Check params.py, where common params are defined. The `use_coral` controls whether or not to use CORAL in pretraining.

## Experiments

Do these experiments as soon as possible:
|   Setting    |  Source Dataset   |          Target Dataset           |          Target Domain Acc | Assignee |
| :----------: | :---------------: | :-------------------------------: | -------------------------: | :------: |
|   ResNet50   |     imagenet      | 16-class-imagenet-val+White Noise |                     47.67% |  Bohua   |
|   ResNet50   | 16-class-imagenet | 16-class-imagenet-val+White Noise |                     25.13% |  Bohua   |
|     ADDA     | 16-class-imagenet | 16-class-imagenet-val+White Noise |           did not converge |  Bohua   |
|    CORAL     | 16-class-imagenet | 16-class-imagenet-val+White Noise |                     73.52% |  Bohua   |
| ADDA + CORAL | 16-class-imagenet | 16-class-imagenet-val+White Noise |                     77.69% |  Bohua   |
|    LeNet     |       MNIST       |               USPS                | 26.34%（decrease from 80%) |    Li    |
|     ADDA     |       MNIST       |               USPS                |                     59.78% |    Li    |
|    CORAL     |       MNIST       |               USPS                |                     54.30% |    Li    |
| ADDA + CORAL |       MNIST       |               USPS                |                     94.56% |    Li    |

Test on 16-class-imagenet-test+White Noise, without any further training. Where setting ResNet50 means ImageNet pretrained, 16-class-imagenet fine tuned;
ResNet50-ImageNet means ImageNet pretrained only; CORAL means ImageNet pretrained, 16-class-imagenet fine tuned with coral loss on 16-class-imagenet-val+white noise; ADDA + CORAL means ImageNet pretrained, 16-class-imagenet fine tuned. All classifier are trained on 16-class-imagenet without other pretraining.
with coral loss and further trained with ADDA on 16-class-imagenet-val+white noise:

|      setting      | Encoder trained on |    Acc | Assignee |
| :---------------: | :----------------: | -----: | :------: |
|     ResNet50      | 16-class-imagenet  |  5.34% |  Bohua   |
| ResNet50-ImageNet |      imagenet      | 13.14% |  Bohua   |
|       CORAL       | 16-class-imagenet  | 38.37% |  Bohua   |
|   ADDA + CORAL    | 16-class-imagenet  | 52.96% |  Bohua   |
