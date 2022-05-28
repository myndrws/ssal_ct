##################################################
# Purpose: Create the splits used in 1%          #
# Author: Amy Andrews                            #
##################################################

# To make sure this was being run correctly
# I ran it in a sort of active monitored loop
# which is why some of the logic is not recursive-
# I was monitoring it as I went to make absolutely sure!


import numpy as np
import pickle
from set_args import get_args
from load_data import images_data, im_transforms
from utils import num_labels_per_class_unequal

with open('splits/kenya80unequalLabelsSplit1.pkl', 'rb') as handle:
    fixed80ids = pickle.load(handle)
with open('splits/kenya1600unequalLabelsSplit1.pkl', 'rb') as handle:
    fixed1600ids = pickle.load(handle)
with open('splits/kenya320unequalLabels1.pkl', 'rb') as handle:
    fixed320ids1 = pickle.load(handle)
with open('splits/kenya320unequalLabels2.pkl', 'rb') as handle:
    fixed320ids2 = pickle.load(handle)
with open('splits/kenya320unequalLabels3.pkl', 'rb') as handle:
    fixed320ids3 = pickle.load(handle)
with open('splits/kenya320unequalLabels4.pkl', 'rb') as handle:
    fixed320ids4 = pickle.load(handle)

# OR
# with open('splits/kenya1600unequalLabelsSplit3.pkl', 'rb') as handle:
#     fixed1600ids = pickle.load(handle)
# with open('splits/kenya80unequalLabelsSplit3.pkl', 'rb') as handle:
#     fixed80ids = pickle.load(handle)

sorted1600ids = np.array(sorted(fixed1600ids))
sorted80ids = np.array(sorted(fixed80ids))

# for the 80 ids plus any additional to remove
growingList = np.unique(sorted(fixed320ids1+fixed320ids2+fixed320ids3+fixed320ids4))

args = get_args()
args['n_available_labels'] = 31775
weak_transform, strong_transform, test_transform = im_transforms(args)
full_train_set = images_data(args, ['train'], weak_transform)  # transform doesn't matter just getting labels


setLen = len(full_train_set.targets)
n_classes = len(np.unique(full_train_set.targets))
all_labels = np.asarray(full_train_set.targets)
rnd = np.random.default_rng(42)
possible_labeled_idx = np.array(np.arange(setLen))
select_from = all_labels[possible_labeled_idx]

subset1600ids = possible_labeled_idx[sorted1600ids]
subset1600targets = select_from[sorted1600ids]
print(np.bincount(subset1600targets), np.bincount(subset1600targets).sum())

subset80ids = possible_labeled_idx[sorted80ids]
subset80targets = select_from[subset80ids]

subsetPreviousids = possible_labeled_idx[growingList]
subsetPrevioustargets = select_from[subsetPreviousids]
print(np.bincount(subsetPrevioustargets), np.bincount(subsetPrevioustargets).sum())

allshouldbein1600 = [i for i in subsetPreviousids if i in subset1600ids]  # this is also correct
assert len(allshouldbein1600) == len(subsetPreviousids)

subsetPreviousremovedfrom1600 = np.array(np.setdiff1d(np.array(subset1600ids), np.array(subsetPreviousids)))
subsetPreviousremovedfrom1600targets = select_from[np.array(np.setdiff1d(np.array(subset1600ids), np.array(subsetPreviousids)))]
print(np.bincount(subsetPreviousremovedfrom1600targets), np.bincount(subsetPreviousremovedfrom1600targets).sum())
# so now should get none of the previous ids
shouldntBeAny = [i for i in subsetPreviousids if i in subsetPreviousremovedfrom1600]  # confirmed
assert len(shouldntBeAny) == 0




###############################################
# create splits of 240 to add to the 80
# using the 1600minus80 pool
###############################################

print(np.bincount(subset80targets))

# the bincounts here have been specifically made so that when adding to the existing 80
# they still roughly match the distribution, whilst also maintaining 320 labels
# done this way so that each of the five splits from the 1520 pool can be unique
# essentially, one entry was removed from index 19, 17 and 16, because there werent enough
# corresponding entires in the full 1520 bincounts remaining to pick a unique example for each split
# therefore these three examples were taken out, but to make up the numbers were added back to indexes
# 15, 5 and 8 - these were the next indices that could have been rounded up
# (according to the num_labels_per_class_unequal function). This all assumes five splits only.

binCounts = np.array([55, 34, 25, 31, 26, 24,  7,  6,  9,  5,  3,  2,  3,  2,  2,  2,  1,  1,  1,  1])
#binCounts = num_labels_per_class_unequal(subsetPreviousremovedfrom1600targets, 240)
labeled_idx = []
for i in range(20):
    idx = np.where(subsetPreviousremovedfrom1600targets == i)[0]
    try:
        idx = rnd.choice(idx, binCounts[i], False)
    except:
        print('error!')
        multFactor = binCounts[i] // len(idx)
        idx = idx.tolist() * multFactor
        if (binCounts[i] - len(idx)) > 0:
            idx = np.concatenate((idx, idx[:(binCounts[i] - len(idx))]), axis=0)
    idx = (np.asarray(subsetPreviousremovedfrom1600)[idx]).tolist()
    for id in idx:
        assert id in subsetPreviousremovedfrom1600
    labeled_idx.extend(idx)

wantToBeFull=[]
for i in labeled_idx:
    if i in subsetPreviousids:
        print('uh oh')
    if i in subset1600ids:
        wantToBeFull.append(i)
assert len(wantToBeFull) == 240

shouldntBeAny = [i for i in labeled_idx if i in subsetPreviousids]
assert len(shouldntBeAny) == 0
totalList = subset80ids.tolist() + labeled_idx
checkunique = np.unique(totalList)
print(np.bincount(select_from[totalList]), np.bincount(select_from[totalList]).sum())
print(np.bincount(select_from[subset80ids]) + np.bincount(select_from[labeled_idx]))  # should be same as line above

with open('splits/kenya320unequalLabelsFrom1600Split3.pkl', 'wb') as f:
    pickle.dump(totalList, f)


#################################
# create 5 train dist splits of 1600
# which do not overlap
# and select 80 from these
#################################

# reprising some code from above at a different time
# to create further splits

args = get_args()
args['n_available_labels'] = 31775
weak_transform, strong_transform, test_transform = im_transforms(args)
full_train_set = images_data(args, ['train'], weak_transform)  # transform doesn't matter just getting labels

setLen = len(full_train_set.targets)
n_classes = len(np.unique(full_train_set.targets))
all_labels = np.asarray(full_train_set.targets)
rnd = np.random.default_rng(42)
possible_labeled_idx = np.array(np.arange(setLen))
select_from_targets = all_labels[possible_labeled_idx]
binCounts1600 = num_labels_per_class_unequal(all_labels, 1600)
binCounts80 = num_labels_per_class_unequal(all_labels, 80)

# existing split
with open('splits/kenya1600unequalLabelsSplit1.pkl', 'rb') as handle:
    split1 = pickle.load(handle)

idsToRemove = split1
rndGen = np.random.default_rng(seed=42)
for i in range(2, 6):

    # this is a sanity check
    subsetTargets = select_from_targets[idsToRemove]
    counts = np.bincount(subsetTargets)
    print(counts)
    assert len(counts) == 20 and counts.sum() == ((i-1) * 1600)

    # create the pool we can search in
    newPoolIds = np.array(np.setdiff1d(np.array(possible_labeled_idx), np.array(idsToRemove)))
    newPoolTargets = select_from_targets[newPoolIds]
    assert len(newPoolIds) == len(newPoolTargets)

    # so now should get none of the previous ids
    assert len([b for b in idsToRemove if b in newPoolIds]) == 0

    # now start selecting the distribution want for the 1600 sets
    # then whilst we have the 1600 ids, select further 80 from that
    labeled_idx_1600 = []
    labeled_idx_80 = []
    for k in range(20):
        idx_for_1600 = np.where(newPoolTargets == k)[0]
        idx_for_1600 = rnd.choice(idx_for_1600, binCounts1600[k], False)
        original_idx_for_1600 = newPoolIds[idx_for_1600]
        for id in original_idx_for_1600:
            assert id in newPoolIds
            assert id not in idsToRemove
        labeled_idx_1600.extend(original_idx_for_1600)

        # repeat for the 80 - then we know these are also included in the 1600
        idx_for_80 = rnd.choice(idx_for_1600, binCounts80[k], False)
        original_idx_for_80 = newPoolIds[idx_for_80]
        for id in original_idx_for_80:
            assert id in labeled_idx_1600
            assert id not in idsToRemove
        labeled_idx_80.extend(original_idx_for_80)

    assert len(np.unique(labeled_idx_1600)) == 1600 and len(np.unique(labeled_idx_80)) == 80
    assert len([b for b in labeled_idx_1600 if b in idsToRemove]) == 0

    print('For 1600:', np.bincount(select_from_targets[labeled_idx_1600]),
          np.bincount(select_from_targets[labeled_idx_1600]).sum())
    print('For 80:', np.bincount(select_from_targets[labeled_idx_80]),
          np.bincount(select_from_targets[labeled_idx_80]).sum())

    # save these with appropriate names
    pathtosave1600 = 'splits/kenya1600unequalLabelsSplit' + str(i) + '.pkl'
    with open(pathtosave1600, 'wb') as f:
        pickle.dump(labeled_idx_1600, f)
    f.close()
    pathtosave80 = 'splits/kenya80unequalLabelsSplit' + str(i) + '.pkl'
    with open(pathtosave80, 'wb') as f:
        pickle.dump(labeled_idx_80, f)
    f.close()

    idsToRemove = idsToRemove + labeled_idx_1600
    assert len(idsToRemove) == (i * 1600)





#################################
# create 5 random splits of 1600
# which do not overlap
#################################

# existing split
with open('splits/kenya1600TrueRandomLabelsSplit1.pkl', 'rb') as handle:
    split1 = pickle.load(handle)

fullset = np.arange(31775)
idsToRemove = split1
rndGen = np.random.default_rng(seed=42)
for i in range(2,6):
    selectfrom = np.array(np.setdiff1d(np.array(fullset), np.array(idsToRemove)))
    pathtosave = 'splits/kenya1600TrueRandomLabelsSplit' + str(i) + '.pkl'
    newlist = rndGen.choice(selectfrom, 1600, replace=False).tolist()
    with open(pathtosave, 'wb') as f:
        pickle.dump(newlist, f)
    f.close()
    for j in newlist:
        assert j not in idsToRemove
    idsToRemove = idsToRemove + newlist
    assert len(idsToRemove) == (i * 1600)
    for k in newlist:
        assert k in idsToRemove
    # select 80 from the 1600 whilst we're here
    pathtosave80 = 'splits/kenya80TrueRandomLabelsSplit' + str(i) + '.pkl'
    subsetofnewlist = rndGen.choice(newlist, 80, replace=False).tolist()
    with open(pathtosave80, 'wb') as f:
        pickle.dump(subsetofnewlist, f)
    f.close()




