import torch
from torch.utils import data
from .image import tiff_to_nd_array, random_augment, normalize_channels, apply_blur
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import os
import pandas as pd

transform = transforms.Compose([])

S2IMAGE_PRE = "s2.20170820"
S2IMAGE_POST = "s2.20170904"

SAR_IMAGE_PRE = "coh_int_20170706_20170823.tif"
SAR_IMAGE_POST = "coh_int_20170823_20170904.tif"

SAR_INTENSITY_PRE = "s1.desc.20170823_123019.tif"
SAR_INTENSITY_POST = "s1.desc.20170904_123019.tif"

MULTITEMP_IMAGE = "S1A_S1_SLC__1SDV_temporal_filtered_2016_2017_T123054_T123118.tif"

VHRIMAGE = "vhr.tif"
LABEL_IMG_DAMAGE = "damage10m.tif"
LABEL_IMG_BUILDINGS = "buildings10m.tif"

class HoustonImageDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir,
                 transform=None,
                 use_multi_sar=False,
                 validation=False,
                 labelimage="buildings10m.tif"):
        self.root_dir = root_dir

        self.n_classes = 2
        self.tile_size = 960
        channel_basis = {'pre_img10': 3, 'post_img10': 3, 'pre_img20': 6, 'post_img20': 6,
                         'pre_sar': 1, 'post_sar': 1, 'multitemp_img': 1,
                         'pre_sar_intensity': 1, 'post_sar_intensity': 1,
                         'vhr': 3}
        channel_dict = dict()

        for item in channel_basis.keys():
            channel_dict['{}'.format(item)] = channel_basis[item]

        self.channels = channel_dict
        self.use_multi_sar = use_multi_sar
        self.validation = validation
        self.labelimage = labelimage


        self.files = []
        for pn in [pn for pn in os.listdir(self.root_dir)
                   if os.path.isdir(os.path.join(self.root_dir, pn))]:
            patch_folder = os.path.join(self.root_dir, pn)

            pre_s2_10_img_path = os.path.join(patch_folder,
                                              S2IMAGE_PRE+".10m.tif")
            post_s2_10_img_path = os.path.join(patch_folder,
                                              S2IMAGE_POST + ".10m.tif")
            pre_s2_20_img_path = os.path.join(patch_folder,
                                              S2IMAGE_PRE + ".20m.tif")
            post_s2_20_img_path = os.path.join(patch_folder,
                                               S2IMAGE_POST + ".20m.tif")
            pre_sar_img_paths = os.path.join(patch_folder,
                                             SAR_IMAGE_PRE)
            post_sar_img_paths = os.path.join(patch_folder,
                                              SAR_IMAGE_POST)
            pre_sar_intensity_img_paths = os.path.join(patch_folder,
                                             SAR_INTENSITY_PRE)
            post_sar_intensity_img_paths = os.path.join(patch_folder,
                                              SAR_INTENSITY_POST)
            multitemp_img_paths = os.path.join(patch_folder,
                                              MULTITEMP_IMAGE)
            vhr_img_path = os.path.join(patch_folder, VHRIMAGE)

            label_img_path = os.path.join(patch_folder, self.labelimage)

            self.files.append({    "sentinel10_img_path_pre": pre_s2_10_img_path,
                                   "sentinel10_img_path_post": post_s2_10_img_path,
                                   "sentinel20_img_path_pre": pre_s2_20_img_path,
                                   "sentinel20_img_path_post": post_s2_20_img_path,
                                   "label_img_path": label_img_path,
                                   "sar_img_paths_pre": [pre_sar_img_paths],
                                   "sar_img_paths_post": [post_sar_img_paths],
                                   "sar_img_paths_pre_intsensity": [pre_sar_intensity_img_paths],
                                   "sar_img_paths_post_intsensity": [post_sar_intensity_img_paths],
                                    "vhr_img_path": vhr_img_path,
                                   "multitemp_img_path": [multitemp_img_paths]})
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def multiclass_generation(self, img):  # FIXME
        for idx in range(len(img)):
            img[idx] = pd.cut(img[idx].flatten(), bins=self.n_classes, labels=False)

    def read_sar_images(self, sar_img_paths, shape):
        def read_single_sar_file(sar_img_path):
            sar = tiff_to_nd_array(sar_img_path).astype(float)
            sar = np.moveaxis(sar, 0, 2)
            sar = cv2.resize(sar, dsize=shape, interpolation=cv2.INTER_LINEAR)
            return sar

        if len(sar_img_paths) == 1:
            sar = read_single_sar_file(sar_img_paths[0])
            return np.expand_dims(sar, axis=0)   # reconstruct (1,320,320)
        else:
            sars = np.array([read_single_sar_file(sar_img_path)
                            for sar_img_path in sar_img_paths])
            sar = np.mean(sars, axis=0)
            return sar

    def read_target_file(self, augmentation, label_img_path):
        label = tiff_to_nd_array(label_img_path).astype(float)
        label = label.clip(min=0)
        label = augmentation(label)
        label = label.squeeze()

        label = torch.from_numpy(label).long()
        return label

    def __getitem__(self, idx):

        def resize(array, shape):
            array = np.moveaxis(array, 0, 2)
            array = cv2.resize(array, dsize=shape, interpolation=cv2.INTER_LINEAR)
            return np.moveaxis(array, 2, 0)

        tile_size = self.tile_size
        h_img10, w_img10 = int(tile_size/10), int(tile_size/10)
        h_sar, w_sar = int(tile_size/5), int(tile_size/5)
        h_vhr, w_vhr = int(tile_size*2/1.25), int(tile_size*2/1.25)

        inputs = dict()
        if self.validation == False:
            augmentation = random_augment()
        else:
            augmentation = random_augment(0)

        # Sentinel 2: B4 (red), B3 (green), B2 (blue), B8 (NIR)
        if 'pre_img10' in self.channels:
            sentinel10_img_path = self.files[idx]['sentinel10_img_path_pre']
            img10 = tiff_to_nd_array(sentinel10_img_path).astype(float)
            img10 = resize(img10, shape=(h_img10, w_img10))

            img10 = augmentation(img10)
            normalize_channels(img10, 10000, 0)

            img10 = torch.from_numpy(img10)
            inputs['pre_img10'] = img10

        if 'post_img10' in self.channels:
            sentinel10_img_path = self.files[idx]['sentinel10_img_path_post']
            img10 = tiff_to_nd_array(sentinel10_img_path).astype(float)
            img10 = resize(img10, shape=(h_img10, w_img10))

            img10 = augmentation(img10)
            normalize_channels(img10, 10000, 0)

            img10 = torch.from_numpy(img10)
            inputs['post_img10'] = img10

        if 'pre_img20' in self.channels:
            sentinel20_img_path = self.files[idx]['sentinel20_img_path_pre']
            img20 = tiff_to_nd_array(sentinel20_img_path).astype(float)

            img20 = resize(img20, shape=(h_img10*2, w_img10*2))
            img20 = augmentation(img20)
            normalize_channels(img20, 10000, 0)

            img20 = torch.from_numpy(img20).float()
            inputs['pre_img20'] = img20


        if 'post_img20' in self.channels:
            sentinel20_img_path = self.files[idx]['sentinel20_img_path_post']
            img20 = tiff_to_nd_array(sentinel20_img_path).astype(float)

            img20 = resize(img20, shape=(h_img10*2, w_img10*2))
            img20 = augmentation(img20)
            normalize_channels(img20, 10000, 0)

            img20 = torch.from_numpy(img20).float()
            inputs['post_img20'] = img20

        if "multitemp_img" in self.channels:
            multitemp_img_path = self.files[idx]['multitemp_img_path']
            sar = self.read_sar_images(multitemp_img_path, (h_sar, w_sar))

            sar = augmentation(sar)
            normalize_channels(sar, 20, -30)

            sar = torch.from_numpy(sar)
            inputs['multitemp_img'] = sar

        # SAR
        if 'pre_sar' in self.channels:
            sar_img_paths = self.files[idx]['sar_img_paths_pre']
            sar = self.read_sar_images(sar_img_paths, (h_sar, w_sar))
            #sar = resize(sar, shape=(h_img10*2, w_img10*2))

            sar = augmentation(sar)
            normalize_channels(sar, 0, 1)

            sar = torch.from_numpy(sar)
            inputs['pre_sar'] = sar

        if 'post_sar' in self.channels:
            sar_img_paths = self.files[idx]['sar_img_paths_post']
            sar = self.read_sar_images(sar_img_paths, (h_sar, w_sar))
            #sar = resize(sar, shape=(h_img10*2, w_img10*2))

            sar = augmentation(sar)
            normalize_channels(sar, 0, 1)

            sar = torch.from_numpy(sar)
            inputs['post_sar'] = sar

        # SAR
        if 'pre_sar_intensity' in self.channels:
            sar_img_paths = self.files[idx]['sar_img_paths_pre_intsensity']
            sar = self.read_sar_images(sar_img_paths, (h_sar, w_sar))
            #sar = resize(sar, shape=(h_img10*2, w_img10*2))

            sar = augmentation(sar)
            normalize_channels(sar, 20, -30)

            sar = torch.from_numpy(sar)
            inputs['pre_sar_intensity'] = sar

        if 'post_sar_intensity' in self.channels:
            sar_img_paths = self.files[idx]['sar_img_paths_post_intsensity']
            sar = self.read_sar_images(sar_img_paths, (h_sar, w_sar))
            #sar = resize(sar, shape=(h_img10*2, w_img10*2))

            sar = augmentation(sar)
            normalize_channels(sar, 20, -30)

            sar = torch.from_numpy(sar)
            inputs['post_sar_intensity'] = sar

        # DigitalGlobe VHR: R G B
        if 'vhr' in self.channels:
            vhr_img_path = self.files[idx]['vhr_img_path']
            img_vhr = tiff_to_nd_array(vhr_img_path).astype(float)
            img_vhr = resize(img_vhr, shape=(int(h_vhr / 2), int(w_vhr / 2)))

            img_vhr = augmentation(img_vhr)
            normalize_channels(img_vhr, 255, 0)
            #img_vhr = resize(img_vhr, shape=(int(h_vhr/2), int(w_vhr/2)))

            img_vhr = torch.from_numpy(img_vhr)
            inputs['vhr'] = img_vhr

        label = self.read_target_file(augmentation, self.files[idx]['label_img_path'])

        tile = os.path.dirname(self.files[idx]["label_img_path"])
        return tile, inputs, (label, )


def train_houston_data_loader(root_dir, batch_size, num_workers, shuffle=True, use_multi_sar=False,
                              validation=False, labelimage="buildings10m.tif"):
    dataset = HoustonImageDataset(root_dir,
                           transform=transform,
                           use_multi_sar=use_multi_sar,
                           validation = validation,
                           labelimage=labelimage)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers)
    return dataloader


def val_houston_data_loader(root_dir, batch_size, num_workers, shuffle=False, use_multi_sar=False,
                            validation=True, labelimage="buildings10m.tif"):
    dataset = HoustonImageDataset(root_dir,
                           transform=transform,
                           use_multi_sar=use_multi_sar,
                           validation=validation,
                           labelimage=labelimage)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers)
    return dataloader

