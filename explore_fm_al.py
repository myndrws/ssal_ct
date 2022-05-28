######################################################
# Purpose: Compare fixmatch active learning results  #
# Author: Amy Andrews                                #
# Resources used:
# UMAP documentation https://umap-learn.readthedocs.io/en/latest/basic_usage.html
# Matplotlib documentation https://matplotlib.org/
##################################################

import os
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import pickle
import numpy as np
from umap import UMAP

from load_data import im_transforms, images_data
from utils import retrieve_fixmatch_outputs, convert_to_names, GenerateResults, vis_ims
from set_args import get_args
from torchvision import transforms

import config

# make sure these are in the same order for labelling
dir_order = ['al_random_basic', 'al_random_rejection', 'al_kcenter_basic',
             'al_kcenter_rejection', 'al_entropy', 'al_entropy_rejection',
             'al_margin_basic', 'al_margin_rejection', 'al_kcenter_hybrid_basic',
             'al_kcenter_hybrid_rejection', 'al_custom_basic', 'al_custom_rejection']

####################################

# get train data for embeddings
names = convert_to_names()
args = get_args()
_, _, test_transform = im_transforms(args)
whole_train_set = images_data(args, ['train'], test_transform, True)
wts = DataLoader(whole_train_set, batch_size=args['batch_size'], shuffle=False)

# get relevant model, generate embeddings, save the dictionary
# GenerateResults generates results for whatever dataset is passed to the method
# Get the very first model which is used to create the initial embeddings
path = args['mod_dir'] + first_mod
kenyaAL = GenerateResults(path)

# retrieve the pseudo labels calculated by this model
# and save these into a file too
ids, pseudo_labels, targets, masks, scores_all, _ = retrieve_fixmatch_outputs(model=kenyaAL.model,
                                                                              dataloader=wts,
                                                                              args=kenyaAL.model_args,
                                                                              p_data=kenyaAL.trainTrackObj.p_data,
                                                                              p_model=kenyaAL.trainTrackObj.p_model)

# generate embeddings
kenyaAL.get_embeddings(wts)

# check everything is in the same order
assert ids.tolist() == kenyaAL.embeddingsDict['context_sheet_ids'].tolist()

# create the 2D umap projections of the embeddings for the training data
umap_2d = UMAP(n_components=2, init='random', random_state=42)
proj_2d = umap_2d.fit_transform(kenyaAL.embeddingsDict['outputs'])

# build dictionary of the newQueries from each AL object using that split
# get the ids of the next 80 examples this model generated queries for
# by referring to the non-rejection custom AL object to get the newQueries at index [1]
# get relevant AL object to see the history of the checkpoints
QueriesFromDirs = {}
for al_dir in dir_order:
    print('Looking in', al_dir, '...')
    dir_path = args['mod_dir'] + al_dir
    all_files = os.listdir(dir_path)
    al_track_objects = [file for file in all_files if 'ActiveLearningTrackObject' in file]
    none_al_track_objects = [file for file in all_files if 'ActiveLearningTrackObject' not in file]
    assert len(all_files) == len(al_track_objects) + len(none_al_track_objects)
    # choosing files without 'proportional_per_class_' deselects files if not 'proportional_per_class[split_num]'
    al_track_object = [file for file in al_track_objects if 'proportional_per_class2_' in file]

    ALobject = torch.load(dir_path + '/' + al_track_object[0], map_location=torch.device('cpu'))
    queriedNext80 = ALobject.newQueries[1]
    QueriesFromDirs[al_dir] = queriedNext80

# mark out the queries that already existed
# get the ids of the 80 initial labels used for this model
with open(splts2, 'rb') as handle:
    alreadyLabelled80 = pickle.load(handle)
alreadyLabelled80 = sorted(alreadyLabelled80)


################

# plot all together
names = np.asarray(convert_to_names())

plt.figure(figsize=(12, 12), dpi=300)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 16
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.scatter(proj_2d[:, 0], proj_2d[:, 1], s=0.25, color="lightsteelblue", label="Unlabelled points")
plt.scatter(proj_2d[alreadyLabelled80, 0], proj_2d[alreadyLabelled80, 1], s=25, marker="*", label="First random labelled examples")
plt.scatter(proj_2d[QueriesFromDirs['al_custom_basic'], 0], proj_2d[QueriesFromDirs['al_custom_basic'], 1], s=25, marker="*", label="Custom strategy queries")
plt.scatter(proj_2d[QueriesFromDirs['al_entropy'], 0], proj_2d[QueriesFromDirs['al_entropy'], 1], s=25, marker="*", label="Entropy strategy queries")

counts = np.zeros(20)
for i in range(len(alreadyLabelled80)):
    species_index = whole_train_set.targets[alreadyLabelled80[i]]
    if (counts[species_index] < 1) or i in [50, 60, 32, 47, 66, 75]:
        plt.annotate(names[species_index], (proj_2d[alreadyLabelled80[i],0], proj_2d[alreadyLabelled80[i],1]), fontsize=14)
    counts[species_index] += 1

lgd=plt.legend(bbox_to_anchor=(0.5, -0.08), loc="lower center", ncol=2, frameon=False)
lgd.legendHandles[0]._sizes = [40]
lgd.legendHandles[1]._sizes = [40]
lgd.legendHandles[2]._sizes = [40]
lgd.legendHandles[3]._sizes = [40]
plt.tight_layout()
filepathname = 'embeddings_first_queries'
plt.savefig('figs/' + filepathname + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight', format='png', dpi=300)
plt.savefig('figs/' + filepathname + '.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight', format='pdf', dpi=300)
plt.show()

#################################
# similarity plots
#################################

# get initial image of an eland
np.bincount(whole_train_set.targets[alreadyLabelled80])
relevant_inds = np.asarray(np.where(whole_train_set.targets[alreadyLabelled80] == 19))
corrected_ind = np.asarray(alreadyLabelled80)[relevant_inds]
eland_path = whole_train_set.im_paths[corrected_ind][0,0]

# get final model for each in custom, kcenter, entropy
from sklearn.metrics import pairwise_distances
dir_order = ['al_custom_basic', 'al_kcenter_basic', 'al_entropy']
models = {}
for al_dir in dir_order:
    print('Looking in', al_dir, '...')
    dir_path = args['mod_dir'] + al_dir
    all_files = os.listdir(dir_path)
    al_track_objects = [file for file in all_files if 'ActiveLearningTrackObject' in file]
    none_al_track_objects = [file for file in all_files if 'ActiveLearningTrackObject' not in file]
    assert len(all_files) == len(al_track_objects) + len(none_al_track_objects)
    # choosing files without 'proportional_per_class_' deselects files if not 'proportional_per_class[split_num]'
    al_track_object = [file for file in al_track_objects if 'proportional_per_class2_' in file]
    ALobject = torch.load(dir_path + '/' + al_track_object[0], map_location=torch.device('cpu'))
    kenyaAL = GenerateResults(ALobject.trainTracksOverAL[-1])
    kenyaAL.get_embeddings(wts)
    orderedMostSimilar = np.argsort(pairwise_distances(X=kenyaAL.embeddingsDict['outputs'][corrected_ind, :].reshape(1, -1),
                                                       Y=kenyaAL.embeddingsDict['outputs'],
                                                       metric="euclidean"))
    models[al_dir] = orderedMostSimilar

with open(splts_eland_entropy, 'wb') as f:
    pickle.dump(models, f)

# now visualise with the other data directory
args['data_dir'] = "/kenya/"
args['im_res'] = 112
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
img = mpimg.imread(args['data_dir'] + eland_path)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 16
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.title("Eland")
plt.imshow(img)
plt.tight_layout()
filepathname = 'anchorEland'
plt.savefig('figs/' + filepathname + '.png', bbox_inches='tight', format='png', dpi=300)
plt.savefig('figs/' + filepathname + '.pdf', bbox_inches='tight', format='pdf', dpi=300)
plt.show()

custom_paths = whole_train_set.im_paths[models['al_custom_basic'].squeeze()][11:21]
custom_targets = whole_train_set.targets[models['al_custom_basic'].squeeze()][11:21]
kcenter_paths = whole_train_set.im_paths[models['al_kcenter_basic'].squeeze()][11:21]
kcenter_targets = whole_train_set.targets[models['al_kcenter_basic'].squeeze()][11:21]
entropy_paths = whole_train_set.im_paths[models['al_entropy'].squeeze()][11:21]
entropy_targets = whole_train_set.targets[models['al_entropy'].squeeze()][11:21]

data_dir=args['data_dir']

fig, axs= plt.subplots(3, 10, figsize=(20, 7))
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
all_ims = custom_paths.tolist() + kcenter_paths.tolist() + entropy_paths.tolist()
all_targets = custom_targets.tolist() + kcenter_targets.tolist() + entropy_targets.tolist()
for ind in range(len(axs.reshape(-1))):
    imtoshow = all_ims[ind]
    labeltitle = names[all_targets[ind]]
    ax = axs.reshape(-1)[ind]
    img = mpimg.imread(data_dir + imtoshow)
    img = transforms.ToTensor()(img)
    img = transforms.Resize((args['im_res'], args['im_res']))(img)
    ax.imshow(img.permute(1, 2, 0))
    ax.set_title(labeltitle)
    ax.set_xticks([])
    ax.set_yticks([])

axs[0,0].set_ylabel('Custom\n')
axs[1,0].set_ylabel('k-Center\n')
axs[2,0].set_ylabel('Entropy\n')
plt.tight_layout()
filepathname = 'elands_11_to_20'
plt.savefig('figs/' + filepathname + '.png', bbox_inches='tight', format='png', dpi=300)
plt.savefig('figs/' + filepathname + '.pdf', bbox_inches='tight', format='pdf', dpi=300)
plt.show()