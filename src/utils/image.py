import numpy as np
# from osgeo import gdal
import rasterio
import cv2


def tiff_to_nd_array(file_path):
    file = rasterio.open(file_path)
    bands = file.read()
    return bands


def random_augment(rate=0.5):
    chance = np.random.rand()
    if chance < rate:
        if chance < rate*(1/3):  # rotation
            angle = np.random.choice([1, 2, 3])

            def augment(img):
                for idx in range(len(img)):
                    channel = img[idx]
                    channel = np.rot90(channel, angle)
                    img[idx] = channel
                return img
        elif chance < rate*(2/3):

            def augment(img):
                for idx in range(len(img)):
                    channel = img[idx]
                    channel = np.flipud(channel)  # horizontal flip
                    img[idx] = channel
                return img
        else:

            def augment(img):
                for idx in range(len(img)):
                    channel = img[idx]
                    channel = np.fliplr(channel)  # vertical flip
                    img[idx] = channel
                return img
    else:
        def augment(img):
            return img

    return augment


def apply_blur(img, type='gauss', ksize=11): # ksize has to be even
    for idx in range(len(img)):
        if type == 'gauss':
            img[idx] = cv2.GaussianBlur(img[idx], (ksize, ksize), 1)
        elif type == 'mean':
            img[idx] = cv2.blur(img[idx], (ksize, ksize))
        elif type == 'dilate':
            kernel = np.ones((ksize, ksize), np.uint8)
            img[idx] = cv2.dilate(img[idx], kernel, iterations=1)
        elif type == 'median':  # FIXME: might not work yet
            img[idx] = cv2.medianBlur(img[idx], cv2.CV_8U)
        elif type == 'laplacian':
            img[idx] = cv2.Laplacian(img[idx], cv2.CV_64F)
        elif type == 'cauchy':
            gamma = 0.2
            img[idx] = 1 / (np.pi * gamma * (1 + ((img[idx] - np.median(img[idx])) / gamma) ** 2))
    return img


def normalize_channels(img, maxval, minval):
    for idx in range(len(img)):
        channel = img[idx]
        channel = channel - minval

        if maxval != 0:
            channel = channel/(maxval-minval)

        img[idx] = channel
    return img



