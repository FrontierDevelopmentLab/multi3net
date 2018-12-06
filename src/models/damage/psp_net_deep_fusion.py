import torch
import math

from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

from models.damage.resnet import resnet50, resnet34, resnet101



from models.damage.resnet import resnet34


class AttentionUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear', align_corners=True)
        return self.conv(p)


class AttentionResNet(nn.Module):
    def __init__(self, model, out_features, batchNorm_momentum=0.1):
        super(AttentionResNet, self).__init__()

        # First Encoder layers
        self.encoder = model
        self.upsamle = AttentionUpsample(out_features, 64)

        # Third Encoder layer after concatenation
        self.s1_s2_conv31 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.s1_s2_bn31 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.s1_s2_conv32 = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.s1_s2_bn32 = nn.BatchNorm2d(2, momentum=batchNorm_momentum)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(1. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        s1_pre = x['pre_sar']
        s1_post = x['post_sar']

        # Stage 1 (S1)
        h, w = 2 * s1_pre.size(2), 2 * s1_pre.size(3)

        s1_pre = F.upsample(input=s1_pre, size=(h, w), mode='bilinear',
                       align_corners=True)
        s1_post = F.upsample(input=s1_post, size=(h, w), mode='bilinear',
                       align_corners=True)

        s1_pre = s1_pre.repeat(1, 3, 1, 1)
        s1_post = s1_post.repeat(1,3,1,1)

        s1_pre_encoded = self.encoder(s1_pre)
        s1_post_encoded = self.encoder(s1_post)

        s1_pre_encoded = self.upsamle(s1_pre_encoded)
        s1_post_encoded = self.upsamle(s1_post_encoded)

        # Difference
        s1 = s1_post_encoded - s1_pre_encoded

        # Stage 2 (S1)
        #s1 = F.relu(self.s1_bn21(self.s1_conv21(s1)))
        s1_s2 = F.relu(self.s1_s2_bn31(self.s1_s2_conv31(s1)))
        s1_s2 = F.relu(self.s1_s2_bn32(
            self.s1_s2_conv32(s1_s2)))

        return s1_s2



def attentionnet34():
    model = resnet34()
    return AttentionResNet(model, out_features=512)


def attentionnet50():
    model = resnet50()
    return AttentionResNet(model, out_features=512*4)


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
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear',
                             align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        #p = F.upsample(input=x, size=(h, w), mode='bilinear', align_corners=True)
        return self.conv(x)


class PSPNet(nn.Module):
    def __init__(self, backend, n_classes=2, sizes=(1, 2, 3, 6), psp_size=2048,
                 deep_features_size=1024,pretrained=True):
        super(PSPNet, self).__init__()
        self.backend = backend
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.train()

        self.attention_net = attentionnet34()

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.LogSoftmax()
        )
        self.final1 = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    @staticmethod
    def _module_parameters(*modules):
        for m in modules:
            for p in m.parameters():
                yield p


    def forward(self, input):
        for item in input:
            if torch.cuda.is_available():
                input['{}'.format(item)] = Variable(input[item].float()).cuda()
            else:
                input['{}'.format(item)] = Variable(input[item].float())

        vhr = Variable(input['vhr'].float()).cuda()

        f = self.backend.forward(vhr)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        p = self.final1(p)

        mask = self.attention_net.forward(input)
        p *= mask

        return self.final(p)


def pspnet():
    model = PSPNet(resnet34(pretrained=True), psp_size=512)
    return model
