from torch.autograd import Variable

from six import text_type
from six import binary_type

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

from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo

def load_weights_sequential(target, source_state):
    new_dict = OrderedDict()
    for (k1, v1), (k2, v2) in zip(target.state_dict().items(), source_state.items()):
        new_dict[k1] = v2
    target.load_state_dict(new_dict)

'''
    Implementation of dilated ResNet-101 with deep supervision. Downsampling is changed to 8x
'''

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, input_channels, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.input_channels = input_channels

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
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
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

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

    def load_imagenet_pretrained(self, url):
        #pretrained_dict = torch.load(model_urls["resnet34"]) # state_dict()
        pretrained_dict = model_zoo.load_url(url)

        own_state = self.state_dict()
        state_dict = dict((k.replace("module.", ""), v)
                          for k, v in own_state.items())
        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            print('following keys are missing and therefore not loaded:')
            print(sorted(missing))

        weights = pretrained_dict["conv1.weight"].repeat(
            1, self.input_channels, 1, 1)[:, :self.input_channels, :, :]
        pretrained_dict["conv1.weight"] = weights

        nn.Module.load_state_dict(
            self, filter_state(own_state, state_dict)
        )


def dilated_resnet18_flexible_input(input_channels, pretrained=True):
    model = ResNet(input_channels, BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_imagenet_pretrained(model_urls["resnet18"])
    return model


def dilated_resnet34_flexible_input(input_channels, pretrained=True):
    model = ResNet(input_channels, BasicBlock, [3, 4, 6, 3])
    if pretrained:
        model.load_imagenet_pretrained(model_urls["resnet34"])
    return model


def dilated_resnet50_flexible_input(input_channels, pretrained=True):
    model = ResNet(input_channels, Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model.load_imagenet_pretrained(model_urls["resnet50"])
    return model

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class InputUpsampleSentinelNet(nn.Module):
    def __init__(self, input_target_size, model, keys, psp_size=512,
                 sizes=(1, 2, 3, 6)):
        super(InputUpsampleSentinelNet, self).__init__()
        self.keys = keys
        self.input_size = input_target_size
        self.p_combined = None

        # First Encoder layers
        self.encoder = model
        self.psp = PSPModule(psp_size, 512, sizes)
        self.drop_2 = nn.Dropout2d(p=0.15)

        self.convp1 = nn.Sequential(
            nn.Conv2d(512, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.convp2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.convp3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(1. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        for item in input:
            if torch.cuda.is_available():
                input['{}'.format(item)] = Variable(input[item].float()).cuda()
            else:
                input['{}'.format(item)] = Variable(input[item].float())

        #x1 = input[self.keys[0]]
        #if "sar" in self.keys[0]:
        #    h, w = 1 * x1.size(2), 4 * x1.size(3)
        #else:
        #    h, w = 8 * x1.size(2), 8 * x1.size(3)

        images = []
        for i in range(len(self.keys)):
            x_i = F.upsample(input=input[self.keys[i]], size= self.input_size,
                               mode='bilinear', align_corners=True)
            images.append(x_i)
        p = torch.cat(images, dim=1)

        p = self.encoder(p)
        p = self.psp(p)

        p = self.convp1(p)
        p = self.drop_2(p)

        p = self.convp2(p)
        p = self.drop_2(p)

        p = self.convp3(p)
        p1 = self.drop_2(p)

        p2 = self.final(p1)

        self.p_combined = torch.cat([p1, p2], dim=1)

        return F.log_softmax(p2, dim=1)

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


def input_upsample_net_34_s2_all():
    keys = ["pre_img10", "post_img10", "pre_img20", "post_img20"]
    in_channels = 4 + 4 + 6 + 6
    model = dilated_resnet34_flexible_input(in_channels)
    input_target_size = (8 * 96, 8 * 96)

    return InputUpsampleSentinelNet(input_target_size, model, keys)


def input_upsample_net_34_s1_all():
    keys = ["pre_sar", "post_sar", "multitemp_img",
            "pre_sar_intensity", "post_sar_intensity"]
    in_channels = 1 * 5
    model = dilated_resnet34_flexible_input(in_channels)
    input_target_size = (8 * 96, 8 * 96)

    return InputUpsampleSentinelNet(input_target_size, model, keys)


def input_upsample_net_34_s1s2():
    keys = ["pre_img10", "post_img10", "pre_img20", "post_img20",
            "pre_sar", "post_sar", "multitemp_img",
            "pre_sar_intensity", "post_sar_intensity"]
    in_channels = 4 + 4 + 6 + 6 + 1 * 5
    input_target_size = (8 * 96, 8 * 96)

    model = dilated_resnet34_flexible_input(in_channels)
    return InputUpsampleSentinelNet(input_target_size, model, keys)

