import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

from six import text_type
from six import binary_type
from collections import OrderedDict

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


# /***************************************************************************
#
#                               SegNetOriginal
#
# ***************************************************************************/
import math
class SegNetOriginal(nn.Module):
    def __init__(self, n_classes, channel_dict, fusion):
        super(SegNetOriginal, self).__init__()
        self.n_classes = n_classes
        self.num_input_channels = 14 #FIXME now uses dictionary

        batchNorm_momentum = 0.1

        self.conv11 = nn.Conv2d(self.num_input_channels, 64,
                                kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn11d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv_final_classes = nn.Conv2d(64, self.n_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(1. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, id1 = F.max_pool2d(
            x12, kernel_size=2, stride=2, return_indices=True)

        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = F.max_pool2d(
            x22, kernel_size=2, stride=2, return_indices=True)

        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, id3 = F.max_pool2d(
            x33, kernel_size=2, stride=2, return_indices=True)

        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, id4 = F.max_pool2d(
            x43, kernel_size=2, stride=2, return_indices=True)

        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, id5 = F.max_pool2d(
            x53, kernel_size=2, stride=2, return_indices=True)

        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = F.relu(self.bn11d(self.conv11d(x12d)))

        x_classes = self.conv_final_classes(x11d)
        if self.n_classes == 1:
            # return F.tanh(x_classes)
            return x_classes
        else:
            return F.log_softmax(x_classes, dim=self.n_classes)

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

    def load_vgg16_weights(self):
        from torchvision.models import vgg16_bn

        model_dict = self.state_dict()
        vgg_dict = vgg16_bn(pretrained=True).state_dict()

        # 64 x 3 x 3
        c = self.num_input_channels
        weights = vgg_dict["features.0.weight"].repeat(
            1, c , 1, 1)[:, :c, :, :]
        model_dict["conv11.weight"] = weights
        model_dict["conv11.bias"] = vgg_dict["features.0.bias"]
        model_dict["bn11.weight"] = vgg_dict["features.1.weight"]
        model_dict["bn11.bias"] = vgg_dict["features.1.bias"]
        model_dict["bn11.running_mean"] = vgg_dict["features.1.running_mean"]
        model_dict["bn11.running_var"] = vgg_dict["features.1.running_var"]

        model_dict["conv12.weight"] = vgg_dict["features.3.weight"]
        model_dict["conv12.bias"] = vgg_dict["features.3.bias"]
        model_dict["bn12.weight"] = vgg_dict["features.4.weight"]
        model_dict["bn12.bias"] = vgg_dict["features.4.bias"]
        model_dict["bn12.running_mean"] = vgg_dict["features.4.running_mean"]
        model_dict["bn12.running_var"] = vgg_dict["features.4.running_var"]

        # 128 x 3 x 3
        model_dict["conv21.weight"] = vgg_dict["features.7.weight"]
        model_dict["conv21.bias"] = vgg_dict["features.7.bias"]
        model_dict["bn21.weight"] = vgg_dict["features.8.weight"]
        model_dict["bn21.bias"] = vgg_dict["features.8.bias"]
        model_dict["bn21.running_mean"] = vgg_dict["features.8.running_mean"]
        model_dict["bn21.running_var"] = vgg_dict["features.8.running_var"]

        model_dict["conv22.weight"] = vgg_dict["features.10.weight"]
        model_dict["conv22.bias"] = vgg_dict["features.10.bias"]
        model_dict["bn22.weight"] = vgg_dict["features.11.weight"]
        model_dict["bn22.bias"] = vgg_dict["features.11.bias"]
        model_dict["bn22.running_mean"] = vgg_dict["features.11.running_mean"]
        model_dict["bn22.running_var"] = vgg_dict["features.11.running_var"]

        # 256 x 3 x 3
        model_dict["conv31.weight"] = vgg_dict["features.14.weight"]
        model_dict["conv31.bias"] = vgg_dict["features.14.bias"]
        model_dict["bn31.weight"] = vgg_dict["features.15.weight"]
        model_dict["bn31.bias"] = vgg_dict["features.15.bias"]
        model_dict["bn31.running_mean"] = vgg_dict["features.15.running_mean"]
        model_dict["bn31.running_var"] = vgg_dict["features.15.running_var"]

        model_dict["conv32.weight"] = vgg_dict["features.17.weight"]
        model_dict["conv32.bias"] = vgg_dict["features.17.bias"]
        model_dict["bn32.weight"] = vgg_dict["features.18.weight"]
        model_dict["bn32.bias"] = vgg_dict["features.18.bias"]
        model_dict["bn32.running_mean"] = vgg_dict["features.18.running_mean"]
        model_dict["bn32.running_var"] = vgg_dict["features.18.running_var"]

        model_dict["conv33.weight"] = vgg_dict["features.20.weight"]
        model_dict["conv33.bias"] = vgg_dict["features.20.bias"]
        model_dict["bn33.weight"] = vgg_dict["features.21.weight"]
        model_dict["bn33.bias"] = vgg_dict["features.21.bias"]
        model_dict["bn33.running_mean"] = vgg_dict["features.21.running_mean"]
        model_dict["bn33.running_var"] = vgg_dict["features.21.running_var"]

        # 512 x 3 x 3
        model_dict["conv41.weight"] = vgg_dict["features.24.weight"]
        model_dict["conv41.bias"] = vgg_dict["features.24.bias"]
        model_dict["bn41.weight"] = vgg_dict["features.25.weight"]
        model_dict["bn41.bias"] = vgg_dict["features.25.bias"]
        model_dict["bn41.running_mean"] = vgg_dict["features.25.running_mean"]
        model_dict["bn41.running_var"] = vgg_dict["features.25.running_var"]

        model_dict["conv42.bias"] = vgg_dict["features.27.bias"]
        model_dict["conv42.weight"] = vgg_dict["features.27.weight"]
        model_dict["bn42.weight"] = vgg_dict["features.28.weight"]
        model_dict["bn42.bias"] = vgg_dict["features.28.bias"]
        model_dict["bn42.running_mean"] = vgg_dict["features.28.running_mean"]
        model_dict["bn42.running_var"] = vgg_dict["features.28.running_var"]

        model_dict["conv43.weight"] = vgg_dict["features.30.weight"]
        model_dict["conv43.bias"] = vgg_dict["features.30.bias"]
        model_dict["bn43.weight"] = vgg_dict["features.31.weight"]
        model_dict["bn43.bias"] = vgg_dict["features.31.bias"]
        model_dict["bn43.running_mean"] = vgg_dict["features.31.running_mean"]
        model_dict["bn43.running_var"] = vgg_dict["features.31.running_var"]

        # 512 x 3 x 3
        model_dict["conv51.weight"] = vgg_dict["features.34.weight"]
        model_dict["conv51.bias"] = vgg_dict["features.34.bias"]
        model_dict["bn51.weight"] = vgg_dict["features.35.weight"]
        model_dict["bn51.bias"] = vgg_dict["features.35.bias"]
        model_dict["bn51.running_mean"] = vgg_dict["features.35.running_mean"]
        model_dict["bn51.running_var"] = vgg_dict["features.35.running_var"]

        model_dict["conv52.weight"] = vgg_dict["features.37.weight"]
        model_dict["conv52.bias"] = vgg_dict["features.37.bias"]
        model_dict["bn52.weight"] = vgg_dict["features.38.weight"]
        model_dict["bn52.bias"] = vgg_dict["features.38.bias"]
        model_dict["bn52.running_mean"] = vgg_dict["features.38.running_mean"]
        model_dict["bn52.running_var"] = vgg_dict["features.38.running_var"]

        model_dict["conv53.bias"] = vgg_dict["features.40.bias"]
        model_dict["conv53.weight"] = vgg_dict["features.40.weight"]
        model_dict["bn53.weight"] = vgg_dict["features.41.weight"]
        model_dict["bn53.bias"] = vgg_dict["features.41.bias"]
        model_dict["bn53.running_mean"] = vgg_dict["features.41.running_mean"]
        model_dict["bn53.running_var"] = vgg_dict["features.41.running_var"]

        self.load_state_dict(model_dict)


def segnet(n_classes=1, num_input_channels=3):
    model = SegNetOriginal(n_classes=n_classes, in_channels=num_input_channels)
    return model
