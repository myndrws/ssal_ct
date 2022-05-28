#################################################################
# Purpose: Implement RandAugment for camera trap images         #
# Author: Amy Andrews                                           #
# Resources used:                                               #
# original RandAugment paper (contains numpy code)              #
# https://arxiv.org/pdf/1909.13719.pdf                          #
# Pytorch documentation                                         #
# https://pytorch.org/vision/stable/transforms.html             #
# unofficial pytorch implementation of randaugment              #
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# official implementation of FixMatch reference to Tensorflow authors at
# https://github.com/google-research/fixmatch/blob/d4985a158065947dba803e626ee9a6721709c570/third_party/auto_augment/augmentations.py
# and lastly unofficial implementation of FixMatch in PyTorch   #
# https://github.com/kekmodel/FixMatch-pytorch/blob/master/dataset/randaugment.py
#################################################################

import torch
import torchvision
from torchvision.transforms import functional
import numpy as np
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from PIL import ImageOps

# first define all possible transformations
# assuming image is a tensor
maxMagnitude = 10

def autocontrast(im, _):
    return functional.autocontrast(im)

def brightness(im, B):
    return functional.adjust_brightness(im, B)

def saturation(im, C):
    return functional.adjust_saturation(im, C)

def contrast(im, C):
    return functional.adjust_contrast(im, C)

def rotate(im, deg):
    if random.random() > 0.5:
        deg = -deg
    return functional.rotate(im, deg)

def sharpness(im, S):
    return functional.adjust_sharpness(im, S)

def solarize(im, T):
    return functional.solarize(im, T)

def identity(im, _):
    return im

def shear_x(im, R):
    if random.random() > 0.5:
        R = -R
    return functional.affine(im, shear=[R, 0], angle=0, scale=1, translate=[0,0])

def shear_y(im, R):
    if random.random() > 0.5:
        R = -R
    return functional.affine(im, shear=[0, R], angle=0, scale=1, translate=[0,0])

def translate_y(im, lmda):
    if random.random() > 0.5:
        lmda = -lmda
    lmda = lmda * im.shape[1]
    return functional.affine(im, translate=[0, int(lmda)], angle=0, scale=1, shear=[0,0])

def translate_x(im, lmda):
    if random.random() > 0.5:
        lmda = -lmda
    lmda = lmda * im.shape[2]
    return functional.affine(im, translate=[int(lmda), 0], angle=0, scale=1, shear=[0,0])

def equalize(im, _):
    toPil = transforms.ToPILImage()
    im = toPil(im)
    return transforms.ToTensor()(functional.equalize(im))

def posterize(im, b):
    toPil = transforms.ToPILImage()
    im = toPil(im)
    im = ImageOps.posterize(im, int(b))
    return transforms.ToTensor()(im)

# define pool for possible transformation
# does not contain transformations that do not make sense
# for camera trap data
def augment_list():

    bccsRange = (0.05, 0.95)  # these are ranges defined in FixMatch
    shearTransRange = (0, 0.3)

    l = [

    (autocontrast, 0, 1),
    (brightness, bccsRange[0], bccsRange[1]),
    (saturation, bccsRange[0], bccsRange[1]),
    (contrast, bccsRange[0], bccsRange[1]),
    (equalize, 0, 1),
    (identity, 0, 1),
    (posterize, 4, 8),
    (rotate, 0, 30),
    (sharpness, bccsRange[0], bccsRange[1]),
    (shear_x, shearTransRange[0], shearTransRange[1]),
    (shear_y, shearTransRange[0], shearTransRange[1]),
    (solarize, bccsRange[0], bccsRange[1]),
    (translate_x, shearTransRange[0], shearTransRange[1]),
    (translate_y, shearTransRange[0], shearTransRange[1]),

    ]

    return l


# define cutout
# which is basically the same as random erasing function
# defined according to FixMatch paper
# 'Sets a random square patch of side-length (L×image width)
# pixels to gray' - this is what RandomErasing does here
def cutout(img, L):
    return torchvision.transforms.RandomErasing(p=1,
                                         scale=(L, L),
                                         ratio=(1, 1),
                                         value=0.5, inplace=False)(img)

# autoaugment function, inc cutout
# autoaugment as trained previously on CIFAR-10 or ImageNet
# view effects https://pytorch.org/vision/master/auto_examples/plot_transforms.html
class AutoAugment:
    def __init__(self, pretrain):
        if pretrain == 'cifar10':
            policy = transforms.AutoAugmentPolicy.CIFAR10
        else:
            policy = transforms.AutoAugmentPolicy.IMAGENET
        self.augmenter = transforms.AutoAugment(policy)
    def __call__(self, im):
        return self.augmenter(im)

# randaugment function
# should set torch manual seed somewhere
# inc cutout
class RandAugment:
    def __init__(self, n, m, cutout=True):
        self.n = n
        self.m = m
        self.maxMagnitude = 10
        self.augment_list = augment_list()
        self.cutout = cutout

    def __call__(self, img):
        self.ops = random.choices(self.augment_list, k=self.n)

        for op, minval, maxval in self.ops:
            # FixMatch authors made the magnitude m random rather than fixed
            # unlike original RandAugment which just used fixed self.m
            val = np.random.randint(1, self.m) * (maxval / self.maxMagnitude)
            img = op(img, val if val >= minval else minval)
        if self.cutout:
            size = np.random.randint(1, self.m) * (0.5 / self.maxMagnitude)
            #size = 0.2
            img = cutout(img, size)

        return img

# test it out
if __name__ == '__main__':

    img = mpimg.imread('')
    plt.imshow(img); plt.show()
    ra = RandAugment(2, 10)
    img = transforms.ToTensor()(img)
    t = ra(img)
    plt.imshow(t.permute(1, 2, 0)); plt.show()