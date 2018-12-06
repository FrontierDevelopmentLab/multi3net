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


class AttentionNetSimple(nn.Module):
    def __init__(self, batchNorm_momentum=0.1):
        super(AttentionNetSimple, self).__init__()

        # First Encoder layers
        self.s1_conv11 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.s1_bn11 = nn.BatchNorm2d(32, momentum=batchNorm_momentum)
        self.s1_conv12 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.s1_bn12 = nn.BatchNorm2d(32, momentum=batchNorm_momentum)

        #self.s2_conv11 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        #self.s2_bn11 = nn.BatchNorm2d(32, momentum=batchNorm_momentum)
        #self.s2_conv12 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        #self.s2_bn12 = nn.BatchNorm2d(32, momentum=batchNorm_momentum)

        # Second Encoder layers after difference
        self.s1_conv21 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.s1_bn21 = nn.BatchNorm2d(32, momentum=batchNorm_momentum)
        self.s1_conv22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.s1_bn22 = nn.BatchNorm2d(32, momentum=batchNorm_momentum)

        #self.s2_conv21 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        #self.s2_bn21 = nn.BatchNorm2d(32, momentum=batchNorm_momentum)
        #self.s2_conv22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        #self.s2_bn22 = nn.BatchNorm2d(32, momentum=batchNorm_momentum)

        # Third Encoder layer after concatenation
        self.s1_s2_conv31 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.s1_s2_bn31 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.s1_s2_conv32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.s1_s2_bn32 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(1. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        #for item in x:
        #    if torch.cuda.is_available():
        #        x['{}'.format(item)] = Variable(x[item].float()).cuda()
        #    else:
        #        x['{}'.format(item)] = Variable(x[item].float())

        s1_pre = x['pre_sar']
        s1_post = x['post_sar']
        s2_pre = x['pre_img10']
        s2_post = x['post_img10']

        # Stage 1 (S1)
        s1_11_pre = F.relu(self.s1_bn11(self.s1_conv11(s1_pre)))
        s1_12_pre = F.relu(self.s1_bn12(self.s1_conv12(s1_11_pre)))

        s1_11_post = F.relu(self.s1_bn11(self.s1_conv11(s1_post)))
        s1_12_post = F.relu(self.s1_bn12(self.s1_conv12(s1_11_post)))

        # Stage 1 (S2)
        #s2_11_pre = F.relu(self.s2_bn11(self.s2_conv11(s2_pre)))
        #s2_12_pre = F.relu(self.s2_bn12(self.s2_conv12(s2_11_pre)))

        #s2_11_post = F.relu(self.s2_bn11(self.s2_conv11(s2_post)))
        #s2_12_post = F.relu(
        #    self.s2_bn12(self.s2_conv12(s2_11_post)))

        # Difference
        s1 = s1_12_post - s1_12_pre
        #s2 = s2_12_post - s2_12_pre

        # Stage 2 (S1)
        s1 = F.relu(self.s1_bn21(self.s1_conv21(s1)))
        #s1 = F.relu(self.s1_bn22(self.s1_conv22(s1)))

        # Stage 2 (S2)
        #s2 = F.relu(self.s2_bn21(self.s2_conv21(s2)))
        #s2 = F.relu(self.s2_bn22(self.s2_conv22(s2)))

        #s1_s2 = torch.cat([s1], dim=1)

        s1_s2 = F.relu(self.s1_s2_bn31(self.s1_s2_conv31(s1)))
        s1_s2 = F.relu(self.s1_s2_bn32(
            self.s1_s2_conv32(s1_s2)))  # TODO: think more, maybe use sigmoid here??

        return s1_s2


# /***************************************************************************
#
#                               Damage Network
#
# ***************************************************************************/
class DamageNetVHR_SimpleFusion(nn.Module):
    def __init__(self, block, layers, n_classes=2, batchNorm_momentum=0.1):
        super(DamageNetVHR_SimpleFusion, self).__init__()
        self.num_in_channels = 3
        self.n_classes = n_classes
        self.inplanes = 64

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
            nn.Conv2d(32, self.n_classes, 1),
        )

        self.attenttion_net = AttentionNetSimple()

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

    def forward(self, input):
        for item in input:
            if torch.cuda.is_available():
                input['{}'.format(item)] = Variable(x[item].float()).cuda()
            else:
                input['{}'.format(item)] = Variable(x[item].float())

        x = input['vhr']

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        attention_mask = self.attenttion_net.forward(input)
        x *= attention_mask

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x4 = F.upsample(x4, (96, 96), mode='bilinear', align_corners=True)
        x5 = self.seg_pre(x4)
        x_classes = self.seg(x5)

        if self.n_classes == 1:
            return x_classes
        else:
            return F.log_softmax(x_classes, dim=1)


def damage_net_vhr_fusion_simple(n_classes):
    model = DamageNetVHR_SimpleFusion(Bottleneck, [3, 4, 6, 3], n_classes=n_classes)
    return model

