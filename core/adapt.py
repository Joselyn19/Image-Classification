"""Adversarial adaptation to train target encoder."""

import os
import time

import torch
import torch.optim as optim
from torch import nn

import params
from utils import cuda_if_possible, get_eta_string


def train_tgt(
    src_encoder, tgt_encoder, critic, src_data_loader, tgt_data_loader
):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    tgt_encoder.train()
    critic.train()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(
        tgt_encoder.parameters(),
        lr=params.c_learning_rate,
        betas=(params.beta1, params.beta2),
    )
    optimizer_critic = optim.Adam(
        critic.parameters(),
        lr=params.d_learning_rate,
        betas=(params.beta1, params.beta2),
    )
    len_data_loader = len(src_data_loader)  # , len(tgt_data_loader))

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs):
        # zip source and target data pair
        # data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        target_iter = iter(tgt_data_loader)
        for step, (images_src, _) in enumerate(src_data_loader):
            time1 = time.time()
            try:
                images_tgt, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(tgt_data_loader)
                images_tgt, _ = next(target_iter)
            ###########################
            # 2.1 train discriminator #
            ###########################

            # make images variable
            images_src = cuda_if_possible(images_src)
            images_tgt = cuda_if_possible(images_tgt)

            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features
            feat_src = src_encoder(images_src)
            feat_tgt = tgt_encoder(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # predict on discriminator
            pred_concat = critic(feat_concat.detach())

            # prepare real and fake label
            label_src = cuda_if_possible(torch.ones(feat_src.size(0)).long())
            label_tgt = cuda_if_possible(torch.zeros(feat_tgt.size(0)).long())
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()

            # optimize critic
            optimizer_critic.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            ############################
            # 2.2 train target encoder #
            ############################

            # zero gradients for optimizer
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()

            # extract and target features
            feat_tgt = tgt_encoder(images_tgt)

            # predict on discriminator
            pred_tgt = critic(feat_tgt)

            # prepare fake labels
            label_tgt = cuda_if_possible(torch.ones(feat_tgt.size(0)).long())

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            optimizer_tgt.step()

            #######################
            # 2.3 print step info #
            #######################
            time2 = time.time()
            if (step + 1) % params.log_step == 0:
                print(
                    (
                        "Epoch [{}/{}] Step [{}/{}]:"
                        + get_eta_string(
                            time1,
                            time2,
                            epoch,
                            params.num_epochs,
                            step,
                            len_data_loader,
                        )
                        + "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
                    ).format(
                        epoch + 1,
                        params.num_epochs,
                        step + 1,
                        len_data_loader,
                        loss_critic.item(),
                        loss_tgt.item(),
                        acc.item(),
                    )
                )

        #############################
        # 2.4 save model parameters #
        #############################
        if (epoch + 1) % params.save_step == 0:
            torch.save(
                critic.state_dict(),
                os.path.join(
                    params.model_root, "ADDA-critic-{}.pt".format(epoch + 1)
                ),
            )
            torch.save(
                tgt_encoder.state_dict(),
                os.path.join(
                    params.model_root,
                    "ADDA-target-encoder-{}.pt".format(epoch + 1),
                ),
            )

    torch.save(
        critic.state_dict(),
        os.path.join(params.model_root, "ADDA-critic-final.pt"),
    )
    torch.save(
        tgt_encoder.state_dict(),
        os.path.join(params.model_root, "ADDA-target-encoder-final.pt"),
    )
    return tgt_encoder
