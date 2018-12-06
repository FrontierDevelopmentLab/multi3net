# full assembly of the sub-parts to form the complete net

from .unet_parts import *

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

from six import text_type
from six import binary_type
from collections import OrderedDict
from models.damage.psp_net_fusion import AttentionNetSimple

class FuseNet(nn.Module):
    def __init__(self, n_classes, channel_dict, fusion, tile_size):
        super(FuseNet, self).__init__()

        self.channel_dict = channel_dict
        self.n_conv_channels = 64
        self.n_classes = n_classes
        self.fusion = fusion
        self.tile_size = tile_size

        # self.final = nn.Conv2d(64, n_classes, kernel_size=1)
        self.softmax = nn.LogSoftmax()

        if self.fusion == 'SVM_base':
            self.S2UpNet10 = S2UpNet10()
            S2_channels = self.S2UpNet10.output_channels

            self.UNet = UNet(S2_channels + self.channel_dict["img10"], n_classes)

        if self.fusion == 'exp':
            self.VHRDownNet5 = VHRDownNet5()
            vhr_channels = self.VHRDownNet5.output_channels

            self.UNet = UNet(vhr_channels, n_classes)

        if self.fusion == 'onlySenTo5m':  # Sen-2 optical up to 5m, fused in Unet with SAR
            n_channels = self.channel_dict["img10"] + self.channel_dict["sar"]
            self.UNet = UNet(n_channels, n_classes)

        if self.fusion == 'BaselineLastWeek':  # All (or subset) Sen-1 and Sen-2 modalities to 10m resolution
            n_channels = self.channel_dict["img10"] + self.channel_dict["img20"] + self.channel_dict["img60"] + self.channel_dict["sar"]
            self.UNet = UNet(n_channels, n_classes)

        if self.fusion == 'BaselineVHR':  # Unet on VHR only (upsampling targets)
            self.UNet = UNet(self.channel_dict["vhr"], n_classes)

        if self.fusion == 'FuseAt5Conv':  # VHR conv to 5m res, S2 20m and 60m res conv to 5m, fuse at 5m
            self.VHRDownNet5 = VHRDownNet5()
            vhr_channels = self.VHRDownNet5.output_channels

            self.S2UpNet5 = S2UpNet5()
            S2_channels = self.S2UpNet5.output_channels

            in_channels = S2_channels + vhr_channels + self.channel_dict["sar"]

            self.UNet = UNet(in_channels, n_classes)

        if self.fusion == 'FuseAt10Conv':  # VHR conv to 10m res, S2 20m and 60m res conv to 10m, fuse at 10m
            self.VHRDownNet10 = VHRDownNet10()
            vhr_channels = self.VHRDownNet10.output_channels

            self.S2UpNet10 = S2UpNet10()
            S2_channels = self.S2UpNet10.output_channels

            in_channels = S2_channels + vhr_channels + self.channel_dict["img10"] + self.channel_dict["sar"]

            self.UNet = UNet(in_channels, n_classes)

        if self.fusion == 'FuseAt10ConvNoVHR':  # S2 20m and 60m res conv to 10m, fuse at 10m

            self.S2UpNet10 = S2UpNet10()
            S2_channels = self.S2UpNet10.output_channels

            in_channels = S2_channels + self.channel_dict["img10"] + self.channel_dict["sar"]

            self.UNet = UNet(in_channels, n_classes)

        if self.fusion == 'FuseAt10Upsample': # VHR downsample to 10m res, S2 20m and 60m res conv to 10m, fuse at 10m
            self.VHRDownNet10 = VHRDownNet10()
            vhr_channels = self.VHRDownNet10.output_channels

            in_channels = vhr_channels + self.channel_dict["img10"] + self.channel_dict["img20"] + self.channel_dict["img60"] + self.channel_dict["sar"]

            self.UNet = UNet(in_channels, n_classes)

        if self.fusion == 'FuseAt5ConvPrePost':  # VHR conv to 5m res, S2 20m and 60m res conv to 5m, fuse at 5m
            self.VHRDownNet5 = VHRDownNet5()
            vhr_channels = self.VHRDownNet5.output_channels

            self.S2UpNet5 = S2UpNet5()
            S2_channels = self.S2UpNet5.output_channels

            self.attention_net = AttentionNetSimple()

            in_channels = S2_channels + vhr_channels

            self.UNet = UNet(in_channels, 64)


    def forward(self, inputs):

        if self.fusion == 'exp':
            x_vhr = self.VHRDownNet5(inputs['vhr'])
            # x = x[:,2,:,:].unsqueeze(1)

            x = self.UNet(x_vhr)

        if self.fusion == 'SVM_base':
            x_s2 = self.S2UpNet10(inputs['img20'], inputs['img60'])

            x_f1 = torch.cat((x_s2, inputs['img10']), 1)

            x = self.UNet(x_f1)

        if self.fusion == 'onlySenTo5m':
            x_10m = nn.functional.upsample(inputs['img10'], size=(int(self.tile_size/5), int(self.tile_size/5)), mode='bilinear')
            x_f1 = torch.cat((x_10m, inputs['sar']), 1)
            x = self.UNet(x_f1)

        if self.fusion == 'BaselineLastWeek':
            size = (int(self.tile_size / 10), int(self.tile_size / 10))
            x_f1 = merge_by_interpolation([inputs['img10'], inputs["img20"], inputs["img60"], inputs['sar']], size=size, mode='bilinear')

            x = self.UNet(x_f1)

        if self.fusion == 'BaselineVHR':
            x = nn.functional.upsample(inputs['vhr'], size=(int(self.tile_size/(2)), int(self.tile_size/(2))), mode='bilinear')

            x = self.UNet(x)

        if self.fusion == 'FuseAt5Conv':
            x_vhr = self.VHRDownNet5(inputs['vhr'])

            x_s2 = self.S2UpNet5(inputs['img10'], inputs['img20'], inputs['img60'])

            x_f1 = torch.cat((x_s2, x_vhr, inputs['sar']), 1)
            x = self.UNet(x_f1)

        if self.fusion == 'FuseAt10Conv':
            x_vhr = self.VHRDownNet10(inputs['vhr'])

            x_s2 = self.S2UpNet10(inputs['img20'], inputs['img60'])

            b, c, h, w = x_s2.shape
            x_sar = torch.nn.functional.upsample(inputs['sar'], size=(h, w), mode='bilinear')

            x_f1 = torch.cat((x_s2, x_vhr, x_sar, inputs['img10']), 1)
            x = self.UNet(x_f1)

        if self.fusion == 'FuseAt10ConvNoVHR':

            x_s2 = self.S2UpNet10(inputs['img20'], inputs['img60'])

            b, c, h, w = x_s2.shape
            x_sar = torch.nn.functional.upsample(inputs['sar'], size=(h, w), mode='bilinear')

            x_f1 = torch.cat((x_s2, x_sar, inputs['img10']), 1)
            x = self.UNet(x_f1)

        if self.fusion == 'FuseAt10Upsample':

            x_vhr = self.VHRDownNet10(inputs['vhr'])

            b, c, h, w = x_vhr.shape
            concatenated = merge_by_interpolation([inputs['sar'], inputs['img10'], inputs['img20'], inputs['img60']], size=(h, w), mode='bilinear')

            x_f1 = torch.cat((x_vhr, concatenated), 1)
            x = self.UNet(x_f1)

        if self.fusion == 'FuseAt5ConvPrePost':
            x_vhr = self.VHRDownNet5(inputs['vhr'])

            x_s2 = self.S2UpNet5(inputs['img10'], inputs['img20'], inputs['img60'])

            x_f1 = torch.cat((x_s2, x_vhr), 1)
            x = self.UNet(x_f1)

            mask = self.attention_net(inputs)

            # b, c, h, w = x.shape
            # mask = torch.nn.functional.upsample(mask, size=(h, w))

            x += mask
            x = self.final(x)


        if self.n_classes == 1:
            return x
        else:
            return self.softmax(x)
            # no softmax + Cross entropy softmax is already implemented in Crossentropy!

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        state_dict = dict((k.replace("module.", ""), v)
                          for k, v in state_dict.items())
        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            print('following keys are missing and therefore not loaded:')
            print(sorted(missing))
        nn.Module.load_state_dict(
            self, filter_state(own_state, state_dict)
        )


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up_skip(1024, 256)
        self.up2 = up_skip(512, 128)
        self.up3 = up_skip(256, 64)
        self.up4 = up_skip(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x

def merge_by_interpolation(list_of_tensors,size=(960,960), mode='bilinear', align_corners=True):

    out=list()
    for tensor in list_of_tensors:
        out.append(nn.functional.upsample(tensor, size, mode=mode,
                                          align_corners=align_corners))

    return torch.cat(out, 1)

class S2UpNet10(nn.Module):
    def __init__(self):
        super(S2UpNet10, self).__init__()
        self.output_channels = 96
        self.up_60_1 = up_nonskip(3, 32, kernel_size=3, stride=3)
        self.up_60_2 = up_nonskip(32, 64)

        self.up_20_1 = up_nonskip(6, 32)

    def forward(self, x20, x60):
        x60_1 = self.up_60_1(x60)
        x60_2 = self.up_60_2(x60_1)

        x20_1 = self.up_20_1(x20)

        x = torch.cat((x60_2, x20_1), 1)

        return x


class S2UpNet5(nn.Module):
    def __init__(self):
        super(S2UpNet5, self).__init__()
        self.output_channels = 64 + 32 + 16
        self.up_60_1 = up_nonskip(3, 16, kernel_size=3, stride=3)
        self.up_60_2 = up_nonskip(16, 32)
        self.up_60_3 = up_nonskip(32, 64)

        self.up_20_1 = up_nonskip(6, 16)
        self.up_20_2 = up_nonskip(16, 32)

        self.up_10_1 = up_nonskip(4, 16)

    def forward(self, x10, x20, x60):
        x60_1 = self.up_60_1(x60)
        x60_2 = self.up_60_2(x60_1)
        x60_3 = self.up_60_3(x60_2)

        x20_1 = self.up_20_1(x20)
        x20_2 = self.up_20_2(x20_1)

        x10_1 = self.up_10_1(x10)

        x = torch.cat((x60_3, x20_2, x10_1), 1)

        return x


class VHRDownNet10(nn.Module):
    def __init__(self):
        super(VHRDownNet10, self).__init__()
        # First 4 layers of VGG16 net # FIXME implement proper VGG net for weight loading
        self.output_channels = 256

        self.down1 = down(3, 32)
        self.down2 = down(32, 64)
        self.down3 = down(64, 128)
        self.down4 = down(128, self.output_channels)

    def forward(self, x):
        vhr_1 = self.down1(x)
        vhr_2 = self.down2(vhr_1)
        vhr_3 = self.down3(vhr_2)
        vhr_4 = self.down4(vhr_3)

        x = vhr_4
        return x


class VHRDownNet5(nn.Module):
    def __init__(self):
        super(VHRDownNet5, self).__init__()
        # First 3 layers of VGG16 net # FIXME implement proper VGG net for weight loading
        self.output_channels = 128

        self.down1 = down(3, 32)
        self.down2 = down(32, 64)
        self.down3 = down(64,  self.output_channels)

    def forward(self, x):
        vhr_1 = self.down1(x)
        vhr_2 = self.down2(vhr_1)
        vhr_3 = self.down3(vhr_2)

        x = vhr_3
        return x



def other_type(s):
    if isinstance(s, text_type):
        return s.encode('utf-8')
    elif isinstance(s, binary_type):
        return s.decode('utf-8')

def try_dicts(k, *ds):
    for d in ds:
        v = d.get(k)
        if v is not None:
            return v
    raise KeyError(k)


def try_types(k, *ds):
    try:
        return try_dicts(k, *ds)
    except KeyError:
        return try_dicts(other_type(k), *ds)


def filter_state(own_state, state_dict):
    return OrderedDict((k, try_types(k, state_dict, own_state))
                       for k in own_state)