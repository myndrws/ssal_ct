##################################################
# Purpose: Load camera trap data for models
# Author: Amy Andrews                            
# Resources used:
# available at https://github.com/omipan/camera_traps_self_supervised/blob/main/datasets.py
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# https://github.com/Celiali/FixMatch/blob/fe55ea9daf353aead58b630da96f357eee728a82/datasets/datasets1.py#L27
##################################################

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Subset
import torch
from torchvision import transforms, datasets
import math
import pickle

from randaugment import RandAugment, AutoAugment
from set_args import get_args
from utils import num_labels_per_class_unequal

class images_data(torch.utils.data.Dataset):

    def __init__(self, args, split, transform, return_single_image=True, indices=None, dataset=None):
        """
        Initialises and stores dataset loading params
        :param args: dict, args for training and running
        :param split: list, split of the data
        :param transform: either single transform or, if return single image is false, tuple of transforms
        :param return_single_image: bool, whether to return single image or weak/strong multiple transforms
        :param indices: list, indices
        :returns torch image dataset ready for passing to DataLoader
        """

        if args['dataset'] == 'kenya':
            da = pd.read_csv(args['metadata'])
            da['im_id'] = np.arange(da.shape[0])

            # bind in the train/test/val split created for this project
            alct_splits = pd.read_csv('splits/kenya_splits.csv')
            da = da.join(alct_splits.set_index('Unnamed: 0'), on='Unnamed: 0')

            # get the class labels before some of the data is excluded
            un_targets, targets = np.unique(da['category_id'].values, return_inverse=True)
            da['category_id_un'] = targets
            self.num_classes = un_targets.shape[0]

            # only use the relevant split's data
            # including if in the ten percent splitting
            da = da[da['img_set2'].isin(split)]

            self.targets = da['category_id_un'].values  # keep as np array
            self.im_paths = da['img_path'].values
            self.original_ids = np.asarray(da['Unnamed: 0'])

            if indices:
                self.targets = self.targets[indices]
                self.im_paths = self.im_paths[indices]
                self.original_ids = self.original_ids[indices]

        elif args['dataset'] == 'serengeti':
            da = pd.read_csv(args['metadata'])
            da['im_id'] = np.arange(da.shape[0])
            da['new_targets_col'] = 500
            species_ordered_like_kenya = ['wildebeest', 'gazelleThomsons', 'zebra', 'cattlePlaceHolder',
                                          'shoatsPlaceHolder', 'impala', 'topi', 'hyenaSpotted', 'warthog',
                                          'giraffe', 'elephant', 'otherBird', 'hare', 'dikDik', 'gazelleGrants',
                                          'jackal', 'mongoose', 'guineaFowl', 'baboon', 'eland']
            for i in range(len(species_ordered_like_kenya)):
                da['new_targets_col'][da['species'] == species_ordered_like_kenya[i]] = i
            da = da[da['new_targets_col'] != 500] # remove any species that aren't our targets
            un_targets, targets = np.unique(da['new_targets_col'].values, return_inverse=True)
            da['category_id_un'] = da['new_targets_col']
            self.num_classes = un_targets.shape[0]
            da = da[da['img_set'].isin(split)]
            self.targets = da['category_id_un'].values  # keep as np array
            self.im_paths = da['img_path'].values
            self.original_ids = np.asarray(da['im_id'])

        elif args['dataset'] == 'cifar':
            self.im_paths = dataset.data[np.asarray(indices)]
            self.targets = (np.asarray(dataset.targets)[np.asarray(indices)])
            self.num_classes = 10
            self.original_ids = indices

        self.num_examples = len(self.im_paths)
        self.transform = transform
        self.data_root = args['data_dir']
        self.return_single_image = return_single_image

    def __len__(self):
        return len(self.im_paths)

    def get_image(self, root_dir, im_path):
        return loader(root_dir + im_path)

    def __getitem__(self, idx):

        op = {}
        op['target'] = self.targets[idx]
        op['id'] = idx
        op['original_id'] = self.original_ids[idx]
        img1_path = self.im_paths[idx]
        op['img1_path'] = img1_path
        if isinstance(img1_path, str):
            img1 = self.get_image(self.data_root, img1_path)
        else:
            img1 = transforms.ToPILImage()(img1_path)

        if self.return_single_image:  # valid/test sets will only ever have single_image = True
            op['im'] = self.transform(img1)
        else:
            # augment_self i.e. two different augmentations of the same image
            assert isinstance(self.transform, tuple), 'If not returning a single image, need a tuple of transforms'
            op['weak_augmentation'] = self.transform[0](img1)
            op['strong_augmentation'] = self.transform[1](img1)

        return op

def loader(im_path_full):
    with open(im_path_full, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# define the kinds of transforms required
def im_transforms(args):

    if args['dataset'] == 'cifar':
        # these transformations are taken from an unofficial replication of FixMatch
        # in pytorch, as the original github/paper was unclear - available at
        # https://github.com/Celiali/FixMatch/blob/fe55ea9daf353aead58b630da96f357eee728a82/datasets/datasets1.py#L27
        assert args['im_res'] == 32, 'CIFAR resolution must be 32x32 in args'
        ls_trans = [transforms.RandomApply([transforms.RandomCrop(size=32, padding=int(32*0.125),
                                                                  padding_mode='reflect'),], p=0.5),
                                              transforms.RandomHorizontalFlip(p=0.5)]
        normalise = [transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])]

    elif args['dataset'] == 'kenya' or args['dataset'] == 'serengeti':
        # ls_trans = [transforms.RandomResizedCrop(args['im_res'], scale=(0.2, 1.0)),
        #         transforms.RandomHorizontalFlip(p=0.5)]
        # UPDATE: RESIZED ALL THE IMAGES MANUALLY TO SIZE 32 SO NOW DON'T NEED RESIZING ON THE FLY
        ls_trans = [transforms.RandomApply([transforms.RandomCrop(size=32, padding=int(32 * 0.125),
                                                                  padding_mode='reflect'), ], p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5)]
        normalise = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    if args['strong_augment'] == 'AutoAugment':
        sa = []
    elif args['strong_augment'] == 'RandAugment':
        sa = [RandAugment(2, 10)]
    elif args['strong_augment'] == 'CTAugment':
        sa = []  # + strong augmentations...
    elif args['strong_augment'] == 'context':
        sa = []
    else:
        sa = []

    weak_transform = transforms.Compose(ls_trans + [transforms.ToTensor()] + normalise)
    strong_transform = transforms.Compose(ls_trans + [transforms.ToTensor()] + sa + normalise)
    test_transform = transforms.Compose([transforms.Resize((args['im_res'], args['im_res']))] +
                                        [transforms.ToTensor()] + normalise)

    return weak_transform, strong_transform, test_transform


# the following function definition is inspired by
# https://github.com/kekmodel/FixMatch-pytorch/blob/master/dataset/cifar.py
# all samples should be used as part of unlabelled set
# expand number of labels by default to fit a single epoch of 1024 iterations
def sampling_data(args, trainset):

    setLen = len(trainset.targets)
    n_classes = len(np.unique(trainset.targets))
    all_labels = np.asarray(trainset.targets)
    rnd = np.random.default_rng(seed=args['seed'])

    # validset ids only required for splitting out cifar
    if args['dataset'] == 'cifar':
        with open('splits/cifarValidIds.pkl', 'rb') as handle:
            valid_idx = pickle.load(handle)
    else:
        valid_idx = None

    if args['dataset'] == 'cifar':
        if args['n_available_labels'] == 10:
            labeled_idx = [4255, 6446, 8580, 11759, 12598, 29349, 29433, 33759, 35345, 38639]
        elif args['n_available_labels'] == 4000:
            with open('splits/cifar4000Labels.pkl', 'rb') as handle:
                labeled_idx = pickle.load(handle)
        elif args['n_available_labels'] == 40:
            with open('splits/cifar40Labels.pkl', 'rb') as handle:
                labeled_idx = pickle.load(handle)

    else:
        assert args['dataset'] == 'kenya'

        if args['n_available_labels'] == 31775:
            labeled_idx = np.array(np.arange(setLen))
            print(np.bincount(all_labels[labeled_idx]), np.bincount(all_labels[labeled_idx]).sum())

        elif args['n_available_labels'] in [1600, 80]:

            if args['labelling_split_type'] == 'equal_per_class':
                assert args['n_available_labels'] == 1600, 'This should only be available for 1600 labels'
                filepath = 'splits/kenya' + str(args['n_available_labels']) + 'equalLabels.pkl'

            elif args['labelling_split_type'] == 'proportional_per_class':
                print('Proportional per class split {} loaded'.format(args['split_id']))
                filepath = 'splits/kenya' + str(args['n_available_labels']) + 'unequalLabelsSplit' + \
                           str(args['split_id']) + '.pkl'

            else:
                assert args['labelling_split_type'] == 'true_uniform_random'
                print('True uniform random split {} loaded'.format(args['split_id']))
                filepath = 'splits/kenya' + str(args['n_available_labels']) + 'TrueRandomLabelsSplit' + \
                           str(args['split_id']) + '.pkl'

            with open(filepath, 'rb') as handle:
                labeled_idx = pickle.load(handle)
            handle.close()
            print(np.unique(all_labels[labeled_idx], return_counts=True), np.bincount(all_labels[labeled_idx]),
                  np.bincount(all_labels[labeled_idx]).sum())

        elif args['n_available_labels'] == 320:
            assert args['labelling_split_type'] == 'proportional_per_class'
            if args['1_perc_subset_id'] in [1,2,3,4,5]:
                filepath = 'splits/kenya' + str(args['n_available_labels']) + \
                           'unequalLabels' + str(args['1_perc_subset_id']) + '.pkl'
            elif args['1_perc_subset_id'] == 6:
                filepath = 'splits/kenya320unequalLabelsFrom1600Split2.pkl'
            else:
                assert args['1_perc_subset_id'] == 7
                filepath = 'splits/kenya320unequalLabelsFrom1600Split3.pkl'
            with open(filepath, 'rb') as handle:
                labeled_idx = pickle.load(handle)
            print('Loading split', args['1_perc_subset_id'], np.bincount(all_labels[labeled_idx]),
                  np.unique(all_labels[labeled_idx], return_counts=True))

        else:
            raise Exception("The data split to initially load is not recognised")

    # expanding the datsets so don't have to re-create iter objects more than necessary
    num_expand_x = math.ceil(args['batch_size'] * args['n_iters_per_epoch'] / args['n_available_labels'])
    labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)]).tolist()
    num_expand_unx = math.ceil((args['batch_size'] * args['fm_ratio_mu']) * args['n_iters_per_epoch'] / setLen)
    unlabeled_idx = np.array(np.arange(setLen))
    unlabeled_idx = np.hstack([unlabeled_idx for _ in range(num_expand_unx)]).tolist()
    rnd.shuffle(labeled_idx)
    rnd.shuffle(unlabeled_idx)

    return labeled_idx, unlabeled_idx, valid_idx


# use this function to load train, valid, test data
def load_data(args):

    if args['dataset'] == 'kenya':
        weak_transform, strong_transform, test_transform = im_transforms(args)
        test_set = images_data(args, ['test'], test_transform)
        val_set = images_data(args, ['valid'], test_transform)

        if args['train_loss'] == 'supervised':
            train_transform = strong_transform if args['strong_augment'] == 'RandAugment' else weak_transform
            train_set = images_data(args, ['train'], train_transform)
            labeled_idx, unlabeled_idx, _ = sampling_data(args, train_set)
            #train_set = Subset(train_set, labeled_idx)
            train_set = images_data(args, split=['train'], transform=train_transform, return_single_image=True,
                                         indices=labeled_idx)
            return train_set, val_set, test_set
        elif args['train_loss'] == 'fixmatch':
            whole_train_set = images_data(args, ['train'], weak_transform, True)
            labeled_idx, unlabeled_idx, _ = sampling_data(args, whole_train_set)
            supervised_set = images_data(args, split=['train'], transform=weak_transform, return_single_image=True,
                                         indices=labeled_idx)
            unsupervised_set = images_data(args, split=['train'], transform=(weak_transform, strong_transform),
                                           return_single_image=False, indices=unlabeled_idx)
            return supervised_set, unsupervised_set, val_set, test_set

    elif args['dataset'] == 'cifar':
        weak_transform, strong_transform, test_transform = im_transforms(args)
        original_train = datasets.CIFAR10(root=args['data_dir'], train=True, download=False, transform=weak_transform)
        labeled_idx, unlabeled_idx, valid_idx = sampling_data(args, original_train)
        val_set = images_data(args, split=None, transform=test_transform, return_single_image=True, indices=valid_idx, dataset=original_train)
        original_test = datasets.CIFAR10(root=args['data_dir'], train=False, download=False, transform=test_transform)
        test_set = images_data(args, split=None, transform=test_transform,
                               return_single_image=True, indices=list(range(10000)), dataset=original_test)

        if args['train_loss'] == 'supervised':
            train_transform = strong_transform if args['strong_augment'] == 'RandAugment' else weak_transform
            train_set = images_data(args, split=None, transform=train_transform, return_single_image=True,
                                    indices=labeled_idx, dataset=original_train)
            return train_set, val_set, test_set
        elif args['train_loss'] == 'fixmatch':
            supervised_set = images_data(args, split=None, transform=weak_transform, return_single_image=True,
                                         indices=labeled_idx, dataset=original_train)
            unsupervised_set = images_data(args, split=None, transform=(weak_transform, strong_transform),
                                           return_single_image=False, indices=unlabeled_idx, dataset=original_train)
            return supervised_set, unsupervised_set, val_set, test_set

    else:
        return 'Dataset not recognised'

# function just to read test set of serengeti data
def get_serengeti_test(args):
    assert args['dataset'] == 'serengeti' and args['im_res'] == 32
    _, _, test_transform = im_transforms(args)
    test_set = images_data(args, split=['test'], transform=test_transform)
    return test_set

###################################################
###################################################

if __name__ == '__main__':

    import set_args
    args = get_args()
    if args['train_loss'] == 'fixmatch':
        supervised_set, unsupervised_set, val_set, test_set = load_data(args)
        print(np.bincount(supervised_set.targets), np.bincount(supervised_set.targets).sum())
    else:
        train_set, val_set, test_set = load_data(args)
        print(np.bincount(train_set.targets), np.bincount(train_set.targets).sum())
