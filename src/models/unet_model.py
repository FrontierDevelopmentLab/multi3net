# full assembly of the sub-parts to form the complete net

from .unet_parts import *

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

from six import text_type
from six import binary_type
from collections import OrderedDict

class UNet(nn.Module):
    def __init__(self, n_classes, channel_dict, fusion):

        super(UNet, self).__init__()

        self.channel_dict = channel_dict
        self.n_conv_channels = 64

        if 'img10' in self.channel_dict:
            self.inc_img10 = inconv(self.channel_dict["img10"], self.n_conv_channels)
        if 'img20' in self.channel_dict:
            self.inc_img20 = inconv(self.channel_dict["img20"], self.n_conv_channels)
        if 'img60' in self.channel_dict:
            self.inc_img60 = inconv(self.channel_dict["img60"], self.n_conv_channels)
        if 'sar' in self.channel_dict:
            self.inc_sar = inconv(self.channel_dict["sar"], self.n_conv_channels)
        if 'vhr' in self.channel_dict:
            self.inc_vhr = inconv(self.channel_dict["vhr"], self.n_conv_channels)

        self.inc = inconv(len(self.channel_dict)*self.n_conv_channels, self.n_conv_channels)  # must ensure data used is specified in options
        self.down_encoder = down(64, 64)
        self.up_encoder = up_nonskip(64, 64)

        if fusion == '/1' or fusion == 'vhr':
            self.down1 = down(64, 128)
            self.down2 = down(128, 256)
            self.down3 = down(256, 512)
            self.down4 = down(512, 512)
            self.up1 = up_skip(1024, 256)
            self.up2 = up_skip(512, 128)
            self.up3 = up_skip(256, 64)
            self.up4 = up_skip(128, 64)
        if fusion == '/2':
            self.down1 = down(64, 128)
            self.down2 = down(128, 256)
            self.down3 = down(256, 256)
            self.up1 = up_skip(512, 128)
            self.up2 = up_skip(256, 64)
            self.up3 = up_skip(128, 64)
        if fusion == '/2^2':
            self.down1 = down(64, 128)
            self.down2 = down(128, 128)
            self.up1 = up_skip(256, 64)
            self.up2 = up_skip(128, 64)
        if fusion == 'seq':
            self.inconv_f1 = inconv(3*self.n_conv_channels, self.n_conv_channels)
            self.inconv_f2 = inconv(2*self.n_conv_channels, self.n_conv_channels)
            self.inc = inconv(2*64, self.n_conv_channels)
            self.down1 = down(64, 128)
            self.down2 = down(128, 256)
            self.down3 = down(256, 256)
            self.up1 = up_skip(512, 128)
            self.up2 = up_skip(256, 64)
            self.up3 = up_skip(128, 64)

        self.outc = outconv(64, n_classes)
        self.n_classes = n_classes
        self.fusion = fusion

    def forward(self, x):

        for item in x:
            if torch.cuda.is_available():
                x['{}'.format(item)] = Variable(x[item].float()).cuda()
            else:
                x['{}'.format(item)] = Variable(x[item].float())

        if 'img10' in x:
            img10_1 = self.inc_img10(x['img10'])
        if 'img20' in x:
            img20_1 = self.inc_img20(x['img20'])
        if 'img60' in x:
            img60_1 = self.inc_img60(x['img60'])
        if 'sar' in x:
            sar_1 = self.inc_sar(x['sar'])
        if 'vhr' in x:
            vhr_1 = self.inc_vhr(x['vhr'])

        concat_input = []

        if self.fusion == '/1':
            if 'img10' in self.channel_dict:
                concat_input.append(img10_1)
            if 'img20' in self.channel_dict:
                img20_2 = self.up_encoder(img20_1)
                concat_input.append(img20_2)
            if 'img60' in self.channel_dict:
                img60_2 = self.up_encoder(img60_1)
                img60_3 = self.up_encoder(img60_2)
                concat_input.append(img60_3)
            if 'sar' in self.channel_dict:
                concat_input.append(sar_1)
            if 'vhr' in self.channel_dict:
                vhr_2 = self.down_encoder(vhr_1)
                vhr_3 = self.down_encoder(vhr_2)
                vhr_4 = self.down_encoder(vhr_3)
                vhr_5 = self.down_encoder(vhr_4)
                concat_input.append(vhr_5)

            x = torch.cat(concat_input, 1)

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

        if self.fusion == '/2':
            if 'img10' in self.channel_dict:
                img10_2 = self.down_encoder(img10_1)
                # img10_3 = self.down_encoder(img10_2)
                concat_input.append(img10_2)
            if 'img20' in self.channel_dict:
                # img20_2 = self.down_encoder(img20_1)
                concat_input.append(img20_1)
            if 'img60' in self.channel_dict:
                img60_2 = self.up_encoder(img60_1)
                concat_input.append(img60_2)
            if 'sar' in self.channel_dict:
                sar_2 = self.down_encoder(sar_1)
                # sar_3 = self.down_encoder(sar_2)
                concat_input.append(sar_2)
            if 'vhr' in self.channel_dict:
                vhr_2 = self.down_encoder(vhr_1)
                vhr_3 = self.down_encoder(vhr_2)
                vhr_4 = self.down_encoder(vhr_3)
                vhr_5 = self.down_encoder(vhr_4)
                vhr_6 = self.down_encoder(vhr_5)
                # vhr_7 = self.down_encoder(vhr_6)
                concat_input.append(vhr_6)

            x = torch.cat(concat_input, 1)

            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)

            x = self.up_encoder(x)
            x = self.outc(x)

        if self.fusion == '/2^2':
            if 'img10' in self.channel_dict:
                img10_2 = self.down_encoder(img10_1)
                img10_3 = self.down_encoder(img10_2)
                concat_input.append(img10_3)
            if 'img20' in self.channel_dict:
                img20_2 = self.down_encoder(img20_1)
                concat_input.append(img20_2)
            if 'img60' in self.channel_dict:
                concat_input.append(img60_1)
            if 'sar' in self.channel_dict:
                sar_2 = self.down_encoder(sar_1)
                sar_3 = self.down_encoder(sar_2)
                concat_input.append(sar_3)
            if 'vhr' in self.channel_dict:
                vhr_2 = self.down_encoder(vhr_1)
                vhr_3 = self.down_encoder(vhr_2)
                vhr_4 = self.down_encoder(vhr_3)
                vhr_5 = self.down_encoder(vhr_4)
                vhr_6 = self.down_encoder(vhr_5)
                vhr_7 = self.down_encoder(vhr_6)
                concat_input.append(vhr_7)

            x = torch.cat(concat_input, 1)

            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x = self.up1(x3, x2)
            x = self.up2(x, x1)

            x = self.up_encoder(x)
            x = self.up_encoder(x)
            x = self.outc(x)

        if self.fusion == 'seq':
            vhr_2 = self.down_encoder(vhr_1)
            vhr_3 = self.down_encoder(vhr_2)
            vhr_4 = self.down_encoder(vhr_3)
            vhr_5 = self.down_encoder(vhr_4)

            f1 = torch.cat([img10_1, sar_1, vhr_5], 1)
            f1_1 = self.inconv_f1(f1)
            f1_2 = self.down_encoder(f1_1)

            f2 = torch.cat([img20_1, f1_2], 1)
            f2_1 = self.inconv_f2(f2)
            # f2_2 = self.down_encoder(f2_1)  # include this if img60(_1/_2) dim is >= 32

            img60_2 = self.up_encoder(img60_1)

            x = torch.cat([img60_2, f2_1], 1)

            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)

            x = self.up_encoder(x)
            x = self.outc(x)

        if self.fusion == 'vhr':
            vhr_2 = self.down_encoder(vhr_1)
            vhr_3 = self.down_encoder(vhr_2)
            vhr_4 = self.down_encoder(vhr_3)
            vhr_5 = self.down_encoder(vhr_4)

            x = vhr_5

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

        if self.n_classes == 1:
            # return F.tanh(x)
            return x
        else:
            # return F.log_softmax(x, dim=self.n_classes)
            return x

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