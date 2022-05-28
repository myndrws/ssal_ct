##################################################
# Purpose: View camera trap image data           #
# Author: Amy Andrews                            #
# Resources used:
# Pytorch documentation https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
##################################################
#%%
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torchvision
import seaborn as sns
import json

from utils import *
from load_data import load_data
from set_args import get_args
from torch.utils.data import DataLoader

import config

args = get_args()

#%%
# look at meta-data
da = pd.read_csv(args['metadata'])
da['im_id'] = np.arange(da.shape[0])
alct_splits = pd.read_csv('splits/kenya_splits.csv')
da = da.join(alct_splits.set_index('Unnamed: 0'), on='Unnamed: 0')
counting(np.asarray(da['daytime']))

with open(animal_dicts, 'rb') as handle:
    animals_dict = json.load(handle)
da['animal_name'] = da['category_id'].apply(lambda x: animals_dict[x]).to_numpy()
sns.countplot(y = "animal_name",
              data = da,
              hue="img_set2",
              order = da["animal_name"].value_counts().index)
plt.tight_layout()
plt.show()

# counts for the subset groups
subset1 = da[da['10_perc_split_train'] == 0]
for i in range(10):
    print(len(da[da['10_perc_split_train'] == i]))

trainOnly = da[da['img_set2']=='train']
counting(np.asarray(subset1['category_id']))
countsDa = da[da['img_set2']=='train'].groupby(['10_perc_split_train', 'category_id']).agg(['count'])


#%%
# look at some train set data
if args['train_loss'] == 'supervised':
    train_set, val_set, test_set = load_data(args)
    train = DataLoader(train_set, batch_size=15, shuffle=False, num_workers=args['num_workers'])
    dataiter = iter(train)
    images = dataiter.next()
    vis_ims(images['img1_path'], images['target'], images['target'], args, nrow=5)


# examine images when fixmatch is set as loss
if args['train_loss'] == 'fixmatch':
    supervised_set, unsupervised_set, val_set, test_set= load_data(args)
    unsupervised = DataLoader(unsupervised_set, batch_size=9, shuffle=False, num_workers=args['num_workers'])
    val = DataLoader(val_set, batch_size=9, shuffle=False, num_workers=args['num_workers'])
    dataiter = iter(unsupervised)
    images = dataiter.next()

    vis_ims(images['img1_path'], images['target'], images['target'], args, nrow=3)
    vis_ims(images['weak_augmentation'], images['target'], images['target'], args, nrow=3)
    vis_ims(images['strong_augmentation'], images['target'], images['target'], args, nrow=3)

    #### view CIFAR data

    import matplotlib.pyplot as plt
    import numpy as np

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataiter = iter(unsupervised)
    imdata = dataiter.next()
    weakims = imdata['weak_augmentation']
    strongims = imdata['strong_augmentation']
    labels = imdata['target']
    imshow_mod(torchvision.utils.make_grid(weakims, nrow=3))
    imshow_mod(torchvision.utils.make_grid(strongims, nrow=3))
    print(' '.join('%5s' % classes[labels[j]] for j in range(9)))

    dataiterval = iter(val)
    imdataval = dataiterval.next()
    valims = imdataval['im']
    labelsval = imdataval['target']
    imshow_mod(torchvision.utils.make_grid(valims, nrow=3))
    print(' '.join('%5s' % classes[labelsval[j]] for j in range(9)))


# examine images in splits
args = get_args()

from load_data import im_transforms, images_data
from torch.utils.data import Subset
weak_transform, strong_transform, test_transform = im_transforms(args)
train_set = images_data(args, ['train'], test_transform)

splitsDict = {}
for i in [1,2,3,4,5]:
    filepath = 'splits/kenya320unequalLabels' + str(i) + '.pkl'
    with open(filepath, 'rb') as handle:
        loaded = pickle.load(handle)
        splitsDict[(i - 1)] = np.asarray(loaded)

train_set = Subset(train_set, splitsDict[0])
#train_set = images_data(args, split=['train'], transform=train_transform, return_single_image=True, indices=labeled_idx)
train = DataLoader(train_set, batch_size=80, shuffle=False, num_workers=args['num_workers'])
dataiter = iter(train)
images = dataiter.next()
vis_ims(images['img1_path'], images['target'], images['target'], args, nrow=10)




#############################
# create plot of zebras
# within-category variation
#############################

imcheck = mpimg.imread('warthog.jpg')
imcheck2 = mpimg.imread('zebra.jpg')
plt.imshow(imcheck2); plt.show()

img1 = mpimg.imread('zebra.jpg')
img2 = mpimg.imread('zebra.jpg')
img3 = mpimg.imread('zebra.jpg')
img4 = mpimg.imread('zebra.jpg')

im1caption = 'Width = {}, Height = {}'.format(img1.shape[1], img1.shape[0])
im2caption = 'Width = {}, Height = {}'.format(img2.shape[1], img2.shape[0])
im3caption = 'Width = {}, Height = {}'.format(img3.shape[1], img3.shape[0])
im4caption = 'Width = {}, Height = {}'.format(img4.shape[1], img4.shape[0])

plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(8, 6))
f, axarr = plt.subplots(2, 2)
axarr[0,0].imshow(img1)
axarr[0,0].set_title("Ideal")
axarr[0,0].axes.yaxis.set_ticks([])
axarr[0,0].axes.xaxis.set_ticks([])
axarr[0,0].set_xlabel(im1caption)

axarr[0,1].imshow(img2)
axarr[0,1].set_title("Poor illumination")
axarr[0,1].axes.yaxis.set_ticks([])
axarr[0,1].axes.xaxis.set_ticks([])
axarr[0,1].set_xlabel(im2caption)

axarr[1,0].imshow(img3)
axarr[1,0].set_title("Perspective change")
axarr[1,0].axes.yaxis.set_ticks([])
axarr[1,0].axes.xaxis.set_ticks([])
axarr[1,0].set_xlabel(im3caption)

axarr[1,1].imshow(img4)
axarr[1,1].set_title("Occlusion")
axarr[1,1].axes.yaxis.set_ticks([])
axarr[1,1].axes.xaxis.set_ticks([])
axarr[1,1].set_xlabel(im4caption)

filepathname ='variationZebra'
plt.tight_layout(pad=2.0)
plt.savefig("figs/" + filepathname + '.png', format='png', dpi=300)
plt.savefig("figs/" + filepathname + '.pdf', format='pdf', dpi=300)
plt.show()


##########################
# topi vs eland
##########################

img1 = mpimg.imread('eland.jpg')
img2 = mpimg.imread('topi.jpg')
img3 = mpimg.imread('thomsons.jpg')
img4 = mpimg.imread('grants.jpg')
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(8, 6))
f, axarr = plt.subplots(2, 2)
axarr[0,0].imshow(img1)
axarr[0,0].set_title("Eland")
axarr[0,0].axes.yaxis.set_ticks([])
axarr[0,0].axes.xaxis.set_ticks([])

axarr[0,1].imshow(img2)
axarr[0,1].set_title("Topi")
axarr[0,1].axes.yaxis.set_ticks([])
axarr[0,1].axes.xaxis.set_ticks([])

axarr[1,0].imshow(img3)
axarr[1,0].set_title("Thomson's Gazelle")
axarr[1,0].axes.yaxis.set_ticks([])
axarr[1,0].axes.xaxis.set_ticks([])

axarr[1,1].imshow(img4)
axarr[1,1].set_title("Grant's Gazelle")
axarr[1,1].axes.yaxis.set_ticks([])
axarr[1,1].axes.xaxis.set_ticks([])

filepathname ='topiElandGazelle'
plt.tight_layout(pad=1.5)
plt.savefig("figs/" + filepathname + '.png', format='png', dpi=300)
plt.savefig("figs/" + filepathname + '.pdf', format='pdf', dpi=300)
plt.show()

##########################
# image stats plot
##########################

plt.scatter(da['width'][da['img_set2']=='train'], da['height'][da['img_set2']=='train'])
plt.scatter(da['width'][da['img_set2']=='test'], da['height'][da['img_set2']=='test'])
plt.show()

