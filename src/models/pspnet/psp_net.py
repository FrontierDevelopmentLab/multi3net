from torch.autograd import Variable

from models.pspnet.resnet import *


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


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_factor):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
        self.upsample_factor = upsample_factor

    def forward(self, x):
        h, w = self.upsample_factor * x.size(2), self.upsample_factor * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear', align_corners=True)
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, backend, n_classes=2, sizes=(1, 2, 3, 6), psp_size=2048,
                 deep_features_size=1024,pretrained=True, upsample_factors=[1,1,1]):
        super(PSPNet, self).__init__()
        self.backend = backend
        self.p_combined = None

        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256, upsample_factor=upsample_factors[0])
        self.up_2 = PSPUpsample(256, 64, upsample_factor=upsample_factors[1])
        self.up_3 = PSPUpsample(64, 64, upsample_factor=upsample_factors[2])

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
#            nn.LogSoftmax()
        )

    @staticmethod
    def _module_parameters(*modules):
        for m in modules:
            for p in m.parameters():
                yield p


    def forward(self, x):
        x = Variable(x['vhr'].float()).cuda()
        f = self.backend.forward(x)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
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

def pspnet_10m():
    model = PSPNet(resnet34(pretrained=True), psp_size=512)
    return model


def pspnet_2m():
    model = PSPNet(resnet34(pretrained=True), psp_size=512,
                   upsample_factors=[2, 2, 1])
    return model


def pspnet_special():
    model = PSPNet(resnet34(pretrained=True), psp_size=512,
                   upsample_factors=[2, 2, 2])
    return model

