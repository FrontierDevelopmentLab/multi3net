from __future__ import print_function, division

from torch import optim
from utils import resume
from utils.trainer import Trainer

from utils import classmetric
import torch.nn as nn
import torch
from torch.utils import data
from utils.image import tiff_to_nd_array
import numpy as np
import torch
import torchvision.transforms as transforms
import os
import pwd
import csv
import pandas as pd
import torch.utils.model_zoo as model_zoo

import os.path as pt
from models.segnet import segnet
from models.unet_model import UNet
from models.fusenet_model import FuseNet
from utils.dataloader import train_data_loader
from utils.dataloader import val_data_loader
from utils.trainer import PolyPolicy
import models

from models.damage.damage_net_vhr import damage_net_vhr
from models.damage.damaged_net_fusion_simple import damage_net_vhr_fusion_simple

from utils import resume
from utils.dataloader_houston import train_houston_data_loader
from utils.dataloader_houston import val_houston_data_loader

from torch.autograd import Variable
import cv2

RESULTS_PATH = os.environ["RESULTS_PATH"]  # "/results"

def main(
        batch_size,
        num_mini_batches,
        nworkers,
        datadir,
        outdir,
        num_epochs,
        snapshot,
        finetune,
        lr,
        n_classes,
        loadvgg,
        network_type,
        fusion,
        data,
):

    n_classes = 2
    tile_size = 960
    channel_basis = {'pre_img10': 3, 'post_img10': 3,
                     'pre_sar': 1, 'post_sar': 1,
                     'vhr': 3}
    channel_dict = dict()
    np.random.seed(0)
    network_type = 'baseline_vhr'

    for item in data:
        channel_dict['{}'.format(item)] = channel_basis[item]



    if network_type == 'baseline_vhr':
        network = damage_net_vhr(n_classes=n_classes)
        network.load_state_dict(model_zoo.load_url(
            'https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    elif network_type == 'baseline_s1':
        network = damage_net_s1(n_classes=n_classes)
        network.load_state_dict(model_zoo.load_url(
            'https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    elif network_type == 'baseline_s2':
        network = damage_net_s2(n_classes=n_classes)
        network.load_state_dict(model_zoo.load_url(
            'https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    elif network_type == 'damagenet_fusion_simple':
        network = damage_net_vhr_fusion_simple(n_classes=n_classes)
        network.load_state_dict(model_zoo.load_url(
            'https://download.pytorch.org/models/resnet50-19c8e357.pth'))

    if pwd.getpwuid(os.getuid())[0] == "jf330":
        finetune = "/Users/jf330/Downloads/results/epoch_{:02}_classes_{:02}.pth".format(
            num_epochs, n_classes)
    elif pwd.getpwuid(os.getuid())[0] == "timrudner":
        finetune = "/Volumes/Data/Google_Drive/AYA_Google_Drive/Git/fdl-eo/code/damage-density-estimation/src/results/epoch_{:02}_classes_{:02}.pth".format(
            num_epochs, n_classes)
    else:
        finetune = "/results/epoch_{:02}_classes_{:02}.pth".format(
            num_epochs, n_classes)

    val = val_houston_data_loader(batch_size=batch_size, num_workers=nworkers,
                                  channels=channel_dict, tile_size=tile_size,
                                  n_classes=n_classes, shuffle=False,
                                  validation=True)
    metric = classmetric()

    if torch.cuda.is_available():
        network = network.cuda()

    if loadvgg == True:
        network.load_vgg16_weights()

    if torch.cuda.is_available():
       network = nn.DataParallel(network).cuda()
    # else:
    #    network = nn.DataParallel(network)

    param_groups = [
        {'params': network.parameters(), 'lr': lr}
    ]

    if finetune or snapshot:
        state = resume(finetune or snapshot, network, None)
    loss_str_list = []

    network.eval()
    for iteration, data in enumerate(val):

        input = data[1]
        input_id = data[0]

        upsample = nn.Upsample(size=(int(tile_size/1.25), int(tile_size/1.25)), mode='bilinear', align_corners=True)  # Harvey
        target = upsample(data[2]["label"])

        if torch.cuda.is_available():
            target = Variable(target.float()).cuda()
        else:
            target = Variable(target.float())

        output_raw = network.forward(input)

        # Normalize
        if n_classes == 1:
            output = output_raw
        else:
            soft = nn.Softmax2d()
            output = soft(output_raw)

        output = upsample(output)

        train_metric = metric(target, output)
        loss_str_list.append("Input ID: {}; Metric: {} ".format(input_id, str(train_metric)))

        # convert zo W x H x C
        if torch.cuda.is_available():
            prediction = output.data.cuda()[0].permute(1, 2, 0)
            target = target.data.cuda()[0].permute(1, 2, 0)
        else:
            prediction = output.data[0].permute(1, 2, 0)
            target = target.data[0].permute(1, 2, 0)

        if not os.path.exists(RESULTS_PATH+"/img"):
            os.makedirs(RESULTS_PATH+"/img")

        # Remove extra dim
        if n_classes == 1:
            prediction_img = prediction.cpu().numpy()
        else:
            prediction_img = np.argmax(prediction, n_classes).cpu().numpy()

        target_img = target.cpu().numpy()

        # Write input image (only first 3 bands)
        # input_img = input.squeeze(0).cpu().numpy()
        #
        # if input_img[:, 0, 0].size >= 3:
        #     input_img = cv2.merge((input_img[0], input_img[1], input_img[2]))
        # else:
        #     input_img = input_img[0]

        #upsample = nn.Upsample(size=(int(tile_size/1.25), int(tile_size/1.25)), mode='bilinear', align_corners=True)  # Harvey

        # cv2.imwrite(RESULTS_PATH+"/img/{}_input_class_{:02}.png".format(iteration, n_classes), input_img*255)
        cv2.imwrite(RESULTS_PATH+"/img/{}_prediction_class_{:02}.png".format(iteration, n_classes), prediction_img*255)
        cv2.imwrite(RESULTS_PATH+"/img/{}_target_class_{:02}.png".format(iteration, n_classes), target_img*255)
        #exit(0)

        with open(RESULTS_PATH + "/MSEloss.csv", "w") as output:
            writer = csv.writer(output, delimiter=';', lineterminator='\n')
            for val in loss_str_list:
                writer.writerow([val])


if __name__ == '__main__':
    import argparse

    text_type=str

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '-b', '--batch-size',
        default=8,
        type=int,
        help='number of images in batch',
    )
    parser.add_argument(
        '-w', '--workers',
        default=1,
        type=int,
        help='number of workers',
    )
    parser.add_argument(
        '-o', '--outdir',
        default='.',
        type=text_type,
        help='output directory',
    )
    parser.add_argument(
        '-e', '--num-epochs',
        default=10,
        type=int,
        help='number of epochs',
    )
    parser.add_argument(
        '-m', '--num-mini-batches',
        default=1,
        type=int,
        help='number of mini batches per batch',
    )
    parser.add_argument(
        '-p', '--datadir',
        default='.',
        type=text_type,
        help='ILSVRC12 dir',
    )
    parser.add_argument(
        '-r', '--resume',
        type=text_type,
        help='snapshot path',
    )
    parser.add_argument(
        '-f', '--finetune',
        type=text_type,
        help='finetune path',
    )
    parser.add_argument(
        '--lr',
        default=0.001,
        type=float,
        help='initial learning rate',
    )
    parser.add_argument(
        '-c', '--n_classes',
        default = 1,
        type = int,
        help = 'prediction type: =1: regression />1: classification',
    )
    parser.add_argument(
        '-x', '--experiment',
        default="PspBuildingsRGB",
        type=text_type,
        help='experiment name',
    )
    parser.add_argument(
        '-v', '--loadvgg',
        default=0,
        type=int,
        help='load vgg16',
    )
    parser.add_argument(
        '-n', '--network_type',
        default='segnet',
        type=text_type,
        help='network type',
    )
    parser.add_argument(
        '-en', '--fusion',
        default='vhr',
        type=text_type,
        help='fusion type',
    )
    parser.add_argument(
        '-d', '--data',
        default=['pre_img10', 'post_img10', 'pre_sar', 'post_sar', 'vhr'],
        type=text_type,
        nargs='+',
        help='datasets used',
    )

    args, unknown = parser.parse_known_args()
    try:
        main(
            args.batch_size,
            args.num_mini_batches,
            args.workers,
            args.datadir,
            args.outdir,
            args.num_epochs,
            args.resume,
            args.finetune,
            args.lr,
            args.n_classes,
            args.loadvgg,
            args.network_type,
            args.fusion,
            args.data,
        )
    except KeyboardInterrupt:
        pass
    finally:
        print()
