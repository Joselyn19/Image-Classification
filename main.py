"""Main script for ADDA."""

import params
from core import eval_src, eval_tgt, train_src, train_tgt, train_src_coral
from models import (
    Discriminator,
    # LeNetClassifier,
    # LeNetEncoder,
    ResNet50Classifier,
    ResNet50Encoder,
)
from utils import get_data_loader, init_model, init_random_seed

if __name__ == "__main__":
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset
    src_data_loader = get_data_loader(params.src_dataset, train=True)
    src_data_loader_eval = get_data_loader(params.src_dataset, train=False)
    tgt_data_loader = get_data_loader(
        params.tgt_dataset, gaussian_blur=True, train=True
    )
    tgt_data_loader_eval = get_data_loader(
        params.tgt_dataset, gaussian_blur=True, train=False
    )

    # load models
    # src_encoder = init_model(net=LeNetEncoder(),
    #                          restore=params.src_encoder_restore)
    src_encoder = init_model(
        net=ResNet50Encoder(),
        restore=params.src_encoder_restore,
        if_init_weights=False,
    )
    src_classifier = init_model(
        net=ResNet50Classifier(),
        restore=params.src_classifier_restore,
        if_init_weights=False,
    )
    tgt_encoder = init_model(
        net=ResNet50Encoder(),
        restore=params.tgt_encoder_restore,
        if_init_weights=False,
    )
    critic = init_model(
        Discriminator(
            input_dims=params.d_input_dims,
            hidden_dims=params.d_hidden_dims,
            output_dims=params.d_output_dims,
        ),
        restore=params.d_model_restore,
    )

    # train source model
    print("=== Training classifier for source domain ===")
    print(">>> Source Encoder <<<")
    print(src_encoder)
    print(">>> Source Classifier <<<")
    print(src_classifier)

    if not (
        src_encoder.restored
        and src_classifier.restored
        and params.src_model_trained
    ):
        if params.use_coral:
            src_encoder, src_classifier = train_src_coral(
                src_encoder, src_classifier, src_data_loader, tgt_data_loader
            )
        else:
            src_encoder, src_classifier = train_src(
                src_encoder, src_classifier, src_data_loader
            )

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_classifier, src_data_loader_eval)

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    print(">>> Target Encoder <<<")
    print(tgt_encoder)
    print(">>> Critic <<<")
    print(critic)

    # init weights of target encoder with those of source encoder
    if not tgt_encoder.restored:
        tgt_encoder.load_state_dict(src_encoder.state_dict())

    if not (
        tgt_encoder.restored and critic.restored and params.tgt_model_trained
    ):
        tgt_encoder = train_tgt(
            src_encoder, tgt_encoder, critic, src_data_loader, tgt_data_loader
        )

    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
    print(">>> domain adaption <<<")
    eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)
