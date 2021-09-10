"""Pre-train encoder and classifier for source dataset."""

import torch.nn as nn
import torch.optim as optim

import params
import time
from models import CORAL
from utils import cuda_if_possible, save_model, get_eta_string


def train_src_coral(encoder, classifier, src_data_loader, tgt_data_loader):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    # setup criterion and optimizer
    optimizer = optim.SGD(
        [
            {"params": encoder.parameters()},
            {"params": classifier.parameters(), "lr": 1e-3},
        ],
        lr=1e-4,
        momentum=0.9
    )
    # optimizer = optim.Adam(
    #     [
    #         {"params": encoder.parameters(), "lr": params.e_learning_rate},
    #         {"params": classifier.parameters(), "lr": params.c_learning_rate},
    #     ],
    #     betas=(params.beta1, params.beta2),
    # )
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs_pre):
        # data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        _lambda = (epoch + 1) / params.num_epochs_pre
        target_iter = iter(tgt_data_loader)
        for step, (src_images, src_labels) in enumerate(src_data_loader):
            try:
                tgt_images, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(tgt_data_loader)
                tgt_images, _ = next(target_iter)

            time1 = time.time()
            # make images and labels variable
            src_images = cuda_if_possible(src_images)
            src_labels = cuda_if_possible(src_labels)
            tgt_images = cuda_if_possible(tgt_images)

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = classifier(encoder(src_images))
            tgt_preds = classifier(encoder(tgt_images))
            coral_loss = CORAL(preds, tgt_preds)
            loss = criterion(preds, src_labels) + _lambda * coral_loss

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            time2 = time.time()
            if (step + 1) % params.log_step_pre == 0:
                print(
                    (
                        "Epoch [{}/{}] Step [{}/{}] "
                        + get_eta_string(
                            time1,
                            time2,
                            epoch,
                            params.num_epochs_pre,
                            step,
                            len(src_data_loader),
                        )
                        + " coral_loss={:.5f} loss={:.5f}"
                    ).format(
                        epoch + 1,
                        params.num_epochs_pre,
                        step + 1,
                        len(src_data_loader),
                        coral_loss.item(),
                        loss.item(),
                    )
                )

        # eval model on test set
        if (epoch + 1) % params.eval_step_pre == 0:
            eval_src(encoder, classifier, src_data_loader)
            encoder.train()
            classifier.train()

        # save model parameters
        if (epoch + 1) % params.save_step_pre == 0:
            save_model(
                encoder, "CORAL-ADDA-source-encoder-{}.pt".format(epoch + 1)
            )
            save_model(
                classifier,
                "CORAL-ADDA-source-classifier-{}.pt".format(epoch + 1),
            )

    # # save final model
    save_model(encoder, "CORAL-ADDA-source-encoder-final.pt")
    save_model(classifier, "CORAL-ADDA-source-classifier-final.pt")

    return encoder, classifier


def train_src(encoder, classifier, data_loader):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=params.c_learning_rate,
        betas=(params.beta1, params.beta2),
    )
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs_pre):
        for step, (images, labels) in enumerate(data_loader):
            time1 = time.time()
            # make images and labels variable
            images = cuda_if_possible(images)
            labels = cuda_if_possible(labels.squeeze_())
            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = classifier(encoder(images))
            loss = criterion(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            time2 = time.time()
            if (step + 1) % params.log_step_pre == 0:
                print(
                    (
                        "Epoch [{}/{}] Step [{}/{}] "
                        + get_eta_string(
                            time1,
                            time2,
                            epoch,
                            params.num_epochs_pre,
                            step,
                            len(data_loader),
                        )
                        + "loss={:.5f}"
                    ).format(
                        epoch + 1,
                        params.num_epochs_pre,
                        step + 1,
                        len(data_loader),
                        loss.item(),
                    )
                )

        # eval model on test set
        if (epoch + 1) % params.eval_step_pre == 0:
            eval_src(encoder, classifier, data_loader)
            encoder.train()
            classifier.train()

        # save model parameters
        if (epoch + 1) % params.save_step_pre == 0:
            save_model(encoder, "ADDA-source-encoder-{}.pt".format(epoch + 1))
            save_model(
                classifier, "ADDA-source-classifier-{}.pt".format(epoch + 1)
            )

    # # save final model
    save_model(encoder, "ADDA-source-encoder-final.pt")
    save_model(classifier, "ADDA-source-classifier-final.pt")

    return encoder, classifier


def eval_src(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = cuda_if_possible(images)
        labels = cuda_if_possible(labels)

        preds = classifier(encoder(images))
        loss += criterion(preds, labels).item()

        pred_cls = preds.data.max(1)[1]
        acc += (pred_cls.eq(labels.data).cpu().sum()).item()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {:.5f}, Avg Accuracy = {:2%}".format(loss, acc))
