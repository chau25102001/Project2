# Modified based on the HRNet repo.

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from lib.datasets.polyp_dataset import PolypDataset
from lib.models.mdeq import MDEQSegNet
import lib.datasets as datasets
from lib.config import config
from lib.config import update_config
from lib.core.seg_criterion import CrossEntropy, OhemCrossEntropy
from lib.core.seg_function import train, validate
from lib.utils.modelsummary import get_model_summary
from lib.utils.utils import create_logger, FullModel, get_rank
from termcolor import colored


def main(config, args):
    final_output_dir = config["OUTPUT_DIR"]
    try:
        os.mkdir(final_output_dir)
    except:
        pass
    cudnn.benchmark = True
    cudnn.deterministic = True
    cudnn.enabled = True
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # build model
    model = MDEQSegNet(config)

    if config["TRAIN"].get("MODEL_FILE"):
        model_state_file = config["TRAIN"]["MODEL_FILE"]
        print(colored('=> loading model from {}'.format(model_state_file), 'red'))

        pretrained_dict = torch.load(model_state_file)
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                           if k[6:] in model_dict.keys()}  # To remove the "model." from state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    torch.cuda.empty_cache()

    train_dataset = PolypDataset(root=args.train_folder, img_path="image",
                                 mask_path="mask")

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["TRAIN"]["BATCH_SIZE_PER_GPU"],
        shuffle=True,
        num_workers=config["WORKERS"],
        pin_memory=True,
        drop_last=True)

    test_size = (config["TEST"]["IMAGE_SIZE"][1], config["TEST"]["IMAGE_SIZE"][0])
    test_dataset = PolypDataset(root=args.test_folder,
                                img_path="images",
                                mask_path="masks")

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["TEST"]["BATCH_SIZE_PER_GPU"],
        shuffle=False,
        num_workers=config["WORKERS"],
        pin_memory=True)

    # criterion
    # if config["LOSS"].get("USE_OHEM"):
    #     criterion = OhemCrossEntropy(ignore_label=config["TRAIN"]["IGNORE_LABEL"],
    #                                  thres=config["LOSS"]["OHEMTHRES"],
    #                                  min_kept=config["LOSS"]["OHEMKEEP"],
    #                                  weight=train_dataset.class_weights)
    # else:
    criterion = nn.BCEWithLogitsLoss()

    model = FullModel(model, criterion)
    model = model.to(device)
    # model(torch.Tensor(1, 3, 512, 512).cuda())

    model = nn.DataParallel(model)

    # optimizer
    if config["TRAIN"]["OPTIMIZER"] == 'sgd':
        optimizer = torch.optim.SGD([{'params':
                                          filter(lambda p: p.requires_grad,
                                                 model.parameters()),
                                      'lr': config["TRAIN"]["LR"]}],
                                    lr=config["TRAIN"]["LR"],
                                    momentum=config["TRAIN"]["MOMENTUM"],
                                    weight_decay=config["TRAIN"]["WD"],
                                    nesterov=config["TRAIN"]["NESTEROV"],
                                    )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = np.int(train_dataset.__len__() /
                         config["TRAIN"]["BATCH_SIZE_PER_GPU"])
    best_mIoU = 0
    last_epoch = 0
    lr_scheduler = None
    if config["TRAIN"]["RESUME"]:
        model_state_file = os.path.join(config["OUTPUT_DIR"], 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file,
                                    map_location=lambda storage, loc: storage)
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            if 'lr_scheduler' in checkpoint:
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, len(trainloader) * config["TRAIN"]["END_EPOCH"], eta_min=1e-6)
                # lr_scheduler.last_epoch = checkpoint['lr_scheduler']['last_epoch']
                # lr_scheduler._step_count = checkpoint['lr_scheduler']['_step_count']
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1e5)
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))

    if lr_scheduler is None:
        if config["TRAIN"]["LR_SCHEDULER"] == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, len(trainloader) * config["TRAIN"]["END_EPOCH"], eta_min=1e-6)
        else:
            assert False, "Shouldn't be here. No LR scheduler!"
            lr_scheduler = None

    start = timeit.default_timer()
    end_epoch = config["TRAIN"]["END_EPOCH"] + config["TRAIN"]["EXTRA_EPOCH"]
    num_iters = config["TRAIN"]["END_EPOCH"] * epoch_iters
    extra_iters = config["TRAIN"]["EXTRA_EPOCH"] * epoch_iters
    writer_dict = {
        'writer': SummaryWriter(config["OUTPUT_DIR"]),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    for epoch in range(last_epoch, end_epoch):
        if epoch >= config["TRAIN"]["END_EPOCH"]:
            train(config, epoch - config["TRAIN"]["END_EPOCH"],
                  config["TRAIN"]["EXTRA_EPOCH"], epoch_iters,
                  config["TRAIN"]["EXTRA_LR"], extra_iters,
                  extra_trainloader, optimizer, lr_scheduler, model,
                  writer_dict, device)
        else:
            train(config, epoch, config["TRAIN"]["END_EPOCH"],
                  epoch_iters, config["TRAIN"]["LR"], num_iters,
                  trainloader, optimizer, lr_scheduler, model, writer_dict,
                  device)

        torch.cuda.empty_cache()
        valid_loss, mean_IoU, IoU_array = validate(config, testloader, model, lr_scheduler, epoch,
                                                   writer_dict, device)
        torch.cuda.empty_cache()
        writer_dict['writer'].flush()

        # if args.local_rank == 0:
        print('=> saving checkpoint to {}'.format(
            final_output_dir + 'checkpoint.pth.tar'))
        torch.save({
            'epoch': epoch + 1,
            'best_mIoU': best_mIoU,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))

        if mean_IoU > best_mIoU:
            best_mIoU = mean_IoU
        torch.save(model.module.state_dict(),
        os.path.join(final_output_dir, 'best.pth'))
        msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
    valid_loss, mean_IoU, best_mIoU)
    print(msg)
    print(IoU_array)

    if epoch == end_epoch - 1:
        torch.save(model.module.state_dict(),
    os.path.join(final_output_dir, 'final_state.pth'))

    writer_dict['writer'].close()
    end = timeit.default_timer()
    print('Hours: %d' % np.int((end - start) / 3600))
    print('Done')

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument("--train_folder", default="/home/s/Gianglt/project_2/DEQ/TrainDataset/TrainDataset", type=str)
        parser.add_argument("--test_folder", default="/home/s/Gianglt/project_2/DEQ/TestDataset/TestDataset/CVC-300",
                        type=str)
        # parser.add_argument("--train_folder", default="", type=str)
        args = parser.parse_args()
        config = yaml.load(open("experiments/polyp/seg_polp_small.yaml", "r"),
        Loader = yaml.FullLoader)
        main(config, args)
