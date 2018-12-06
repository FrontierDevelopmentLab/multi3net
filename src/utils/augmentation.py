
import torch
import numpy as np


class ToTensorTarget(object):

    def __call__(self, sample):
        sat_img, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        return {'image': torch.from_numpy(sat_img.transpose(2,0,1)),
                'label': torch.from_numpy(label).float().div(255).long()}


