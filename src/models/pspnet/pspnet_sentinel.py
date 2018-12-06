from torch.autograd import Variable

from models.pspnet.resnet import *

class SentinelPSPNet(nn.Module):
    def __init__(self, model, keys, inchannels, batchNorm_momentum=0.1):
        super(SentinelPSPNet, self).__init__()
        self.keys = keys

        # First Encoder layers
        self.encoder = model
        self.conv_pre = nn.Sequential(
            nn.Conv2d(inchannels, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Conv2d(8, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.PReLU()
        )
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
            nn.LogSoftmax()
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

        x1 = input[self.keys[0]]
        if "sar" in self.keys[0]:
            h, w = 4 * x1.size(2), 4 * x1.size(3)
        else:
            h, w = 8 * x1.size(2), 8 * x1.size(3)

        images = []
        for i in range(len(self.keys)):
            x_i = F.upsample(input=input[self.keys[i]], size= (h, w),
                               mode='bilinear', align_corners=True)
            images.append(x_i)

        p = torch.cat(images, dim=1)
        p = self.conv_pre(p)

        p = self.encoder(p)

        p = self.convp1(p)
        p = self.drop_2(p)

        p = self.convp2(p)
        p = self.drop_2(p)

        p = self.convp3(p)
        p = self.drop_2(p)

        return self.final(p)


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


def psp34_sentinel2_all():
    model = resnet34()
    keys = ["pre_img10", "post_img10", "pre_img20", "post_img20"]
    in_channels = 4 + 4 + 6 + 6
    return SentinelPSPNet(model, keys, in_channels)


def psp34_sentinel1_all():
    model = resnet34()
    keys = ["pre_sar", "post_sar", "multitemp_img",
            "pre_sar_intensity", "post_sar_intensity"]
    in_channels = 1 * 5
    return SentinelPSPNet(model, keys, in_channels)


def psp34_sentinel1_and_sentinel2():
    model = resnet34()
    keys = ["pre_img10", "post_img10", "pre_img20", "post_img20",
            "pre_sar", "post_sar", "multitemp_img",
            "pre_sar_intensity", "post_sar_intensity"]
    in_channels = 4 + 4 + 6 + 6 + 1 * 5
    return SentinelPSPNet(model, keys, in_channels)

