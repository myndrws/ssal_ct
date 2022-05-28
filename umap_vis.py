##################################################
# Purpose: Build projections for vis with UMAP   #
# Author: Amy Andrews                            #
# Resources used:
# UMAP documentation
# https://umap-learn.readthedocs.io/en/latest/basic_usage.html
# https://umap-learn.readthedocs.io/en/latest/plotting.html#plotting-larger-datasets
##################################################

from load_data import load_data, im_transforms, images_data
from utils import *
from set_args import get_args

from torch.utils.data import DataLoader
import pickle
import numpy as np
import pandas as pd

from umap import UMAP

import config

# get train data for embeddings
names = convert_to_names()
args = get_args()
_, _, test_transform = im_transforms(args)
whole_train_set = images_data(args, ['train'], test_transform, True)
wts = DataLoader(whole_train_set, batch_size=args['batch_size'], shuffle=False,
                 num_workers=args['num_workers'], pin_memory=True)

# get relevant AL object to see the history of the checkpoints
ALpath = alpathumap
ALobject = torch.load(ALpath)

# for demonstration purposes in the app
# get the model checkpointed at trained with 80 labels, and retrieve the next 80 labels that would be selected
# WITHOUT REJECTION SAMPLING; rejection sampling assumes interaction whereas no one has interacted with the app yet
# because the initial model was not included in the list the path is given below
# as this was the 'base pretraining' model used across all the AL experiments

# get the ids of the 80 initial labels used for this model
with open('splits/kenya80unequalLabelsSplit1.pkl', 'rb') as handle:
    alreadyLabelled80 = pickle.load(handle)
alreadyLabelled80 = sorted(alreadyLabelled80)
# get the ids of the next 80 examples this model generated queries for
# by referring to the non-rejection custom AL object to get the newQueries at index [1]
queriedNext80 = ALobject.newQueries[1]

# get relevant model, generate embeddings, save the dictionary
# GenerateResults generates results for whatever dataset is passed to the method
# Get the very first model which is used to create the initial embeddings
path = firstmodinit
save_name = 'customQuerying80LabsAt132Epochs'
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

# create 3D projections
umap_3d = UMAP(n_components=3, init='random', random_state=42)
proj_3d = umap_3d.fit_transform(kenyaAL.embeddingsDict['outputs'])

# create a mask for the targets that are actually available
# a single col for Label type
# key is 0 for pseudo, 1 for query, 2 for true
pseudoQueryTrueMask = np.zeros(len(ids))
pseudoQueryTrueMask[queriedNext80] = 1
pseudoQueryTrueMask[alreadyLabelled80] = 2
assert len(pseudoQueryTrueMask[pseudoQueryTrueMask==1]) == 80
assert len(pseudoQueryTrueMask[pseudoQueryTrueMask==2]) == 80

# create a column for the amalgamation of pseudo and true labels
# by which to colour the points according to species
pseudoOrTrueLabels = []
double_checking = []
for i in range(len(ids)):
    if pseudoQueryTrueMask[i] == 0 or pseudoQueryTrueMask[i] == 1:
        pseudoOrTrueLabels.append(pseudo_labels[i])
    else:
        double_checking.append(i)
        pseudoOrTrueLabels.append(targets[i])
assert len(double_checking) == len(pseudoQueryTrueMask[pseudoQueryTrueMask==2])
assert double_checking == alreadyLabelled80#.tolist()

# create just a queries-or-not column for two sizes
queriesOrNot = np.zeros(len(ids))
queriesOrNot[queriedNext80] = 100
queriesOrNot[queriesOrNot!=100] = 10

# true targets mixed with the word 'unknown'
true_targets = []
for i in range(len(ids)):
    if pseudoQueryTrueMask[i] == 0 or pseudoQueryTrueMask[i] == 1:
        true_targets.append("Unknown")
    else:
        true_targets.append(names[targets[i]])
true_targets = np.asarray(true_targets)
assert len(true_targets[true_targets != 'Unknown']) == 80

# add everything together in a single dataframe for opening in the dashboard
embedding_info = kenyaAL.embeddingsDict
da = pd.read_csv("kenya_context_final.csv")
night_day = da['daytime'][da['Unnamed: 0'].isin(embedding_info['context_sheet_ids'])]
yesNoNightDay = np.where(night_day==0, 'Night, dusk, dawn', 'Day, dusk, dawn')
pqt = ['Pseudo-label', 'Suggested for labelling', 'True label']
scatter_data = pd.DataFrame({'index': np.arange(len(embedding_info['context_sheet_ids'])),
                             'x': proj_2d[:, 0],
                             'y': proj_2d[:, 1],

                             'x3d': proj_3d[:, 0],
                             'y3d': proj_3d[:, 1],
                             'z3d': proj_3d[:, 2],

                             'PseudoAndTrueAmalg': pseudoOrTrueLabels,
                             'Label Type': [pqt[int(i)] for i in pseudoQueryTrueMask],
                             'QueriesOrNot': queriesOrNot,

                             'ID': embedding_info['context_sheet_ids'],
                             'True Label': true_targets,
                             'Pseudo-label': [names[i] for i in pseudo_labels],
                             'Species': [names[i] for i in pseudoOrTrueLabels],
                             'Daytime': yesNoNightDay,
                             'Image Path': embedding_info['im_paths']})

scatter_data["AnimalsLabels"] = scatter_data["Species"]
scatter_data["AnimalsLabels"][scatter_data['Label Type'] == 'Suggested for labelling'] = "Suggested Query"
scatter_data["AnimalsLabels"][scatter_data['Label Type'] == 'True label'] = "True Label"
scatter_data["HoverName"] = scatter_data["Label Type"] + ": " + scatter_data["Species"]
scatter_data["HoverName"][scatter_data['Label Type'] == 'Suggested for labelling'] = scatter_data["HoverName"] + "?"

# finally save it all for reading into the app
scatter_data.to_csv("embs/scatterDataFrame_" + save_name + ".csv", index=False)