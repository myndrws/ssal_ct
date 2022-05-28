######################################################
# Purpose: Compare supervised vs fixmatch results    #
# Author: Amy Andrews                                #
# Resources used:
# Matplotlib documentation https://matplotlib.org/
##################################################

from utils import GenerateResults, vis_ims
from load_data import load_data, images_data, im_transforms
from set_args import get_args

import numpy as np
import pickle
import torch
import os
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import config

args = get_args()
supervised_set, unsupervised_set, val_set, test_set = load_data(args)
test = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'])

# kenya, supervised, all labels
kenyaSupAllLabs = GenerateResults(kenyaSupAllLabs)

# kenya, sup, 1600, equal dist
kenyaSup1600Equal = GenerateResults(kenyaSup1600Equal)
# kenya, FM, 1600, equal dist
kenyaFM1600Equal = GenerateResults(kenyaFM1600Equal)

# kenya, sup, 1600, unequal dist
kenyaSup1600Unequal = GenerateResults(kenyaSup1600Unequal)
# kenya, FM, 1600, unequal dist
kenyaFM1600Unequal = GenerateResults(kenyaFM1600Unequal)
# kenya, FM+DA, 1600, unequal dist
kenyaFMDA1600Unequal = GenerateResults(kenyaFMDA1600Unequal)


# kenya, sup, 80, unequal dist
kenyaSup80Unequal = GenerateResults(kenyaSup80Unequal)
# kenya, FM+DA, 80, unequal dist
kenyaFMDA80Unequal = GenerateResults(kenyaFMDA80Unequal)

# kenya, sup, 320, unequal dist, splits 1-5
kenyaSup320split1 = GenerateResults(kenyaSup320split1)
kenyaSup320split2 = GenerateResults(kenyaSup320split2)
kenyaSup320split3 = GenerateResults(kenyaSup320split3)
kenyaSup320split4 = GenerateResults(kenyaSup320split4)
kenyaSup320split5 = GenerateResults(kenyaSup320split5)

# kenya, fm, 320, unequal dist, splits 1-5
kenyaFMDA320split1 = GenerateResults(kenyaFMDA320split1)
kenyaFMDA320split2 = GenerateResults(kenyaFMDA320split2)
kenyaFMDA320split3 = GenerateResults(kenyaFMDA320split3)
kenyaFMDA320split4 = GenerateResults(kenyaFMDA320split4)
kenyaFMDA320split5 = GenerateResults(kenyaFMDA320split5)

# also read in the alternative 1600 and 320 splits
# kenya, FM+DA, 1600, split2, unequal dist - it's PT because of the power outage half way through
kenyaFMDA1600Unequalsplit2 = GenerateResults(kenyaFMDA1600Unequalsplit2)
# kenya, FM+DA, 1600, split3, unequal dist
kenyaFMDA1600Unequalsplit3 = GenerateResults(kenyaFMDA1600Unequalsplit2)

# kenya, FM+DA, 320, split2, unequal dist
kenyaFMDA320Unequalsplit6 = GenerateResults(kenyaFMDA320Unequalsplit6)
# kenya, FM+DA, 320, split3, unequal dist
kenyaFMDA320Unequalsplit7 = GenerateResults(kenyaFMDA320Unequalsplit7)


all_models = [kenyaSupAllLabs, kenyaSup1600Equal, kenyaFM1600Equal, kenyaSup1600Unequal, kenyaFM1600Unequal,
              kenyaFMDA1600Unequal, kenyaSup80Unequal, kenyaFMDA80Unequal, kenyaSup320split1, kenyaSup320split2,
              kenyaSup320split3, kenyaSup320split4, kenyaSup320split5, kenyaFMDA320split1, kenyaFMDA320split2,
              kenyaFMDA320split3, kenyaFMDA320split4, kenyaFMDA320split5, kenyaFMDA1600Unequalsplit2,
              kenyaFMDA1600Unequalsplit3, kenyaFMDA320Unequalsplit6, kenyaFMDA320Unequalsplit7]
mod_names = ['kenyaSupAllLabs', 'kenyaSup1600Equal', 'kenyaFM1600Equal', 'kenyaSup1600Unequal', 'kenyaFM1600Unequal',
              'kenyaFMDA1600Unequal', 'kenyaSup80Unequal', 'kenyaFMDA80Unequal', 'kenyaSup320split1', 'kenyaSup320split2',
              'kenyaSup320split3', 'kenyaSup320split4', 'kenyaSup320split5', 'kenyaFMDA320split1', 'kenyaFMDA320split2',
              'kenyaFMDA320split3', 'kenyaFMDA320split4', 'kenyaFMDA320split5', 'kenyaFMDA1600Unequalsplit2',
              'kenyaFMDA1600Unequalsplit3', 'kenyaFMDA320Unequalsplit6', 'kenyaFMDA320Unequalsplit7']

class saveResultsOnly():

    def __init__(self, GenResultsObj, name):
        self.name = name
        self.path = GenResultsObj.mod_path
        self.test_acc = GenResultsObj.test_accuracy
        self.test_top5_acc = GenResultsObj.test_top5_accuracy
        self.test_f1 = GenResultsObj.test_f1
        self.test_precision = GenResultsObj.classReport['macro avg']['precision']
        self.test_recall = GenResultsObj.classReport['macro avg']['recall']
        self.test_acc_by_class = GenResultsObj.testAccuracyByClass
        self.cmat = GenResultsObj.cmat
        self.test_f1_prec_recall_by_class = GenResultsObj.classReport

for m in range(len(all_models)):
    all_models[m].accuracy_breakdowns(test, produce_plots=False)
    pkl_name = 'results/passive_mods/' + mod_names[m] + '_kenya_test_results.pkl'
    newObjToSave = saveResultsOnly(all_models[m], mod_names[m])
    with open(pkl_name, 'wb') as handle:
        pickle.dump(newObjToSave, handle, pickle.HIGHEST_PROTOCOL)
    handle.close()

# generate results on train for the fmda80 model
weak_transform, strong_transform, test_transform = im_transforms(args)
train_set = images_data(args, ['train'], weak_transform)
train = DataLoader(train_set, batch_size=32, shuffle=False)
kenyaFMDA80UnequalReloaded = GenerateResults(kenyaFMDA80UnequalReloaded)
kenyaFMDA80UnequalReloaded.accuracy_breakdowns(train)
kenyaFMDA80UnequalReloadedObjToSave = saveResultsOnly(kenyaFMDA80UnequalReloaded, 'fmda80_train')
with open(splts80unequal, 'wb') as handle:
    pickle.dump(kenyaFMDA80UnequalReloadedObjToSave, handle, pickle.HIGHEST_PROTOCOL)
handle.close()
