import torch
import math

from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

from six import text_type
from six import binary_type
from collections import OrderedDict
from torchvision.models.resnet import Bottleneck

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
#                               Damage Network
#
# ***************************************************************************/
class DamageNetVHR(nn.Module):
    def __init__(self, block, layers, n_classes=2, batchNorm_momentum=0.1):
        super(DamageNetVHR, self).__init__()
        self.num_in_channels = 3
        self.inplanes = 64
        self.num_classes = n_classes
        self.n_classes = 2

        self.conv1 = nn.Conv2d(self.num_in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        self.seg_pre = nn.Sequential(
            nn.Conv2d(2048, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),
        )

        self.seg = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(32, self.num_classes, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(1. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

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

    def forward(self, x):
        for item in x:
            if torch.cuda.is_available():
                x['{}'.format(item)] = Variable(x[item].float()).cuda()
            else:
                x['{}'.format(item)] = Variable(x[item].float())

        x = x['vhr']

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.seg_pre(x4)

        x_classes = self.seg(x5)
        x_classes = F.upsample(x_classes, (96, 96), mode='bilinear',
                               align_corners=True) #FIXME: do corretcly

        if self.n_classes == 1:
            return x_classes
        else:
            return F.log_softmax(x_classes, dim=1)


def damage_net_vhr(n_classes):
    model = DamageNetVHR(Bottleneck, [3, 4, 6, 3])
    return model

