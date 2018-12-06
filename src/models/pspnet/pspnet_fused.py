from torch.autograd import Variable

from models.pspnet.resnet import *


class PSPNetFused(nn.Module):
    def __init__(self, backend_hr, backend_sentinel, n_classes=2):
        super(PSPNetFused, self).__init__()
        self.backend_hr = backend_hr
        self.backend_sentinel = backend_sentinel

        self.final = nn.Sequential(
            nn.Conv2d(64 * 2 + 2 * 2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.LogSoftmax()
        )

    def parameters(self):
        for n, p in self.named_parameters():
            if "final" in n:
                yield p


    @staticmethod
    def _module_parameters(*modules):
        for m in modules:
            for p in m.parameters():
                yield p

    def forward(self, x):
        hr_preds_buildings = self.backend_hr.forward(x)
        lr_preds_s2_buildings = self.backend_sentinel.forward(x)

        hr_features = self.backend_hr.p_combined
        lr_features_s2 = self.backend_sentinel.p_combined

        x = torch.cat([hr_features, lr_features_s2], dim=1)
        pred_damage = self.final(x)

        return hr_preds_buildings, lr_preds_s2_buildings, pred_damage


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

    """ 
    def load_pretrained(self):
        #path_hr = "/models_vhr/best_model_buildings.pth"
        path_hr = "/results/damage_hr/epoch_20_classes_02.pth"
        path_sen = "/results/damage_s1s2/epoch_14_classes_02.pth"
        snapshot = torch.load(path_hr)
        model_state_hr = snapshot.pop('model_state', snapshot)

        snapshot = torch.load(path_sen)
        model_state_sen = snapshot.pop('model_state', snapshot)

        psp_hr = pspnet()
        #psp_sen = psp34_sentinel1_and_sentinel2()

        psp_hr.load_state_dict(model_state_hr)
        psp_sen.load_state_dict(model_state_sen)
        self.backend_hr = psp_hr
        self.backend_sentinel = psp_sen
    """

from models.pspnet.psp_net import *
from models.low_res_seg.input_upsample import input_upsample_net_34_s2_all
from models.low_res_seg.input_upsample import input_upsample_net_34_s1_all


def pspnet_fused_s2_10m():
    psp_hr = pspnet_10m()
    psp_sen = input_upsample_net_34_s2_all()

    model = PSPNetFused(psp_hr, psp_sen)
    return model


def pspnet_fused_s1_10m():
    psp_hr = pspnet_10m()
    psp_sen = input_upsample_net_34_s1_all()

    model = PSPNetFused(psp_hr, psp_sen)
    return model


def pspnet_fused_s2_2m():
    psp_hr = pspnet_2m()
    psp_sen = input_upsample_net_34_s2_all()

    model = PSPNetFused(psp_hr, psp_sen)
    return model
