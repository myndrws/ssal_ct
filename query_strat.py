##################################################
# Purpose: Existing and custom query strategies  #
# Author: Amy Andrews                            #
# Resources used:
# https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py
# https://github.com/microsoft/CameraTraps/tree/f69de8339bf19806fbe943963e2b385b948c35b2/research/active_learning
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
# These are referenced below
##################################################


# imports
import copy
import numpy as np
import math
import torch
import torch.nn as nn
import pickle
from sklearn.metrics import pairwise_distances
from scipy import stats

from torch.utils.data import DataLoader
from load_data import im_transforms, images_data
from utils import load_model, PMovingAverage, retrieve_fixmatch_outputs
from ema import EMA


# class that takes args as argument
# saves the list of existing ids and then the new list of ids at each iteration
# i.e. for every AL iteration saves the trainTrack object from the model, the labelled ids (and their pseudolabels)
class active_learning_querying(object):

    def __init__(self, trainTrackObj, allIdsLen, queryStrategy, nQueriesPerIter, argsForThisRun):

        assert queryStrategy in ['random', 'kcenter', 'kcenter_hybrid',
                                 'custom', 'least_certain', 'margin',
                                 'predetermined_1600_proportional', 'predetermined_1600_true_rand'], \
            'Query strategy not recognised'

        cppath = trainTrackObj.args['mod_path']
        self.al_tracking_path = cppath.replace(".pt", "") + "ActiveLearningTrackObject.pt"

        assert trainTrackObj.args['n_available_labels'] == 80

        if argsForThisRun['labelling_split_type'] == 'true_uniform_random':
            idspath = 'splits/kenya80TrueRandomLabelsSplit' + str(argsForThisRun['split_id']) + '.pkl'
        else:
            idspath = 'splits/kenya' + str(argsForThisRun['n_available_labels']) + 'unequalLabelsSplit' + \
                           str(argsForThisRun['split_id']) + '.pkl'

        with open(idspath, 'rb') as handle:
            self.startingLabels = pickle.load(handle)
        handle.close()
        self.startingLabels = sorted(self.startingLabels)

        # labelledIds should be the list of labelledIds starting with WITHOUT the expansion
        self.trainTracksOverAL = [trainTrackObj]
        self.labelledIdsOverAL = [self.startingLabels]
        self.queryStrategy = queryStrategy
        self.allIdsLen = allIdsLen
        self.rndGen = np.random.default_rng(seed=42)
        self.nQueriesPerALIter = nQueriesPerIter
        self.argsForThisRun = argsForThisRun
        self.newQueries = [self.startingLabels]  # these count as queries at iter1
        self.nCycledThrough = []

    def add_new_traintrack(self, trainTrackObj):
        self.trainTracksOverAL.append(trainTrackObj)

    def query_selection(self):
        # takes in selection of existing example ids
        # excludes these from next possible choices
        # selects n next choices using query selection
        # saves the new queries as well as the cumulative list of queries
        # and returns the new examples to label as well as the new examples plus the already labelled ones
        # concatenated together and expanded for running through with the model
        latestLabelledIds = np.unique(sorted(self.labelledIdsOverAL[-1]))
        batchSize = self.argsForThisRun['batch_size']
        itersPerEpoch = self.argsForThisRun['n_iters_per_epoch']

        possible_labeled_idx = np.array(np.arange(self.allIdsLen))

        subsetPreviousids = possible_labeled_idx[latestLabelledIds]
        assert np.all(subsetPreviousids.tolist() == latestLabelledIds.tolist())
        subsetPreviousRemoved = np.array(np.setdiff1d(np.array(possible_labeled_idx), np.array(subsetPreviousids)))

        if self.queryStrategy == 'random':
            if 'use_rejection_sampling' in self.argsForThisRun and self.argsForThisRun['use_rejection_sampling']:
                newQueries = self.random_with_rejection_sampling()
            else:
                newQueries = self.rndGen.choice(subsetPreviousRemoved, self.nQueriesPerALIter, replace=False).tolist()
        elif self.queryStrategy == 'kcenter' or self.queryStrategy == 'kcenter_hybrid':
            # the if/else for rejection sampling happens within the kcenter function
            # the hybrid vs normal kcenter also decided within function
            newQueries = self.kcenter()
        elif self.queryStrategy == 'custom':
            # the if/else for rejection sampling happens within the custom querying function
            newQueries = self.custom_querying()
        elif self.queryStrategy == 'least_certain' or self.queryStrategy == 'margin':
            # the if/else for rejection sampling happens within the least certain function
            # entropy vs margin based also decided within function
            newQueries = self.least_certain()
        elif self.queryStrategy == 'predetermined_1600_proportional':
            # selects new queries from a pool where they've already been chosen before the AL procedure
            # using the splits which follow the train distribtion (80 is a subset of final 1600)
            idspath = 'splits/kenya1600unequalLabelsSplit' + str(self.argsForThisRun['split_id']) + '.pkl'
            newQueries = self.choose_from_predetermined_random(idspath)
        elif self.queryStrategy == 'predetermined_1600_true_rand':
            # selects new queries from a pool where they've already been chosen before the AL procedure
            # using the splits which are 'true random' (80 is a subset of final 1600)
            idspath = 'splits/kenya1600TrueRandomLabelsSplit' + str(self.argsForThisRun['split_id']) + '.pkl'
            newQueries = self.choose_from_predetermined_random(idspath)
        else:
            raise Exception

        self.newQueries.append(newQueries)
        # appended to old queries
        totalIdsForNextIter = np.unique(np.concatenate(self.newQueries, axis=0)).astype(int)
        print('AL loop: length of new queries is', len(self.newQueries[-1]))
        assert len(totalIdsForNextIter) == (
                    len(latestLabelledIds) + len(newQueries)), 'The ids are apparently not unique'
        self.labelledIdsOverAL.append(totalIdsForNextIter)

        # expanded for returning to use with dataloader
        num_expand_x = math.ceil(batchSize * itersPerEpoch / len(totalIdsForNextIter))
        expandedTotalIdsForNextIter = np.hstack([totalIdsForNextIter for _ in range(num_expand_x)]).tolist()
        self.rndGen.shuffle(expandedTotalIdsForNextIter)

        # return the ids of the examples to be loaded in dataloader
        return expandedTotalIdsForNextIter

    def kcenter(self):

        # ADAPTED FROM https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py
        # based on theory and code provided by
        # @inproceedings{sener2018active,
        #     title={Active Learning for Convolutional Neural Networks: A Core-Set Approach},
        #     author={Ozan Sener and Silvio Savarese},
        #     booktitle={International Conference on Learning Representations},
        #     year={2018},
        #     url={https://openreview.net/forum?id=H1aIuk-RW},
        # }

        # takes in existing labelled ids
        # create the embeddings for all the training data using existing embedding function
        # use a function to use these to update distances to centers
        # in a for loop, for each id, get the index with the argmax distances of min distances
        # append the index to newQueries and rerun the function that updates distances to centers
        # return batch of new queries
        lastTTobj = self.trainTracksOverAL[-1]
        modForKcenter, _ = load_model(lastTTobj.args)

        if self.trainTracksOverAL[-1].args['ema']:
            lastEMAmod = EMA(modForKcenter, 0.999)
            modForKcenter.load_state_dict(lastTTobj.model)
            lastEMAmod.load_state_dict(lastTTobj.ema_model)
            lastEMAmod.apply_shadow()
        else:
            modForKcenter.load_state_dict(lastTTobj.model)

        # get the embeddings and corresponding ids for all the training data
        ids, embeddings = get_embeddings_for_querying(modForKcenter, self.argsForThisRun)
        del modForKcenter, lastEMAmod
        print('Al loop: successfully retrieved train embeddings')

        # if using rejection sampling then we'll also need the pseudo-labels and targets of these
        if ('use_rejection_sampling' in self.argsForThisRun and self.argsForThisRun['use_rejection_sampling']) or \
                (self.queryStrategy == 'kcenter_hybrid'):
            original_model, _ = load_model(lastTTobj.args)

            if self.trainTracksOverAL[-1].args['ema']:
                lastEMAmod = EMA(original_model, 0.999)
                original_model.load_state_dict(lastTTobj.model)
                lastEMAmod.load_state_dict(lastTTobj.ema_model)
                lastEMAmod.apply_shadow()
            else:
                original_model.load_state_dict(lastTTobj.model)

            try:
                p_data = lastTTobj.p_data
                p_model = lastTTobj.p_model
            except:
                print('AL loop: No p_data or p_model recorded, setting up new bland objects...')
                p_data = torch.as_tensor(np.repeat(1 / 20, 20), device=self.argsForThisRun['device'])
                p_model = PMovingAverage(nclass=20, buf_size=128, device=self.argsForThisRun['device'])

            ids, pseudo_labels, targets, masks, scores, _ = get_pseudo_labels_and_masks(original_model,
                                                                                     p_data=p_data,
                                                                                     p_model=p_model,
                                                                                     args=self.argsForThisRun)
            del original_model, p_data, p_model, lastEMAmod, _, scores

        # get the initial centers based on the existing queries and embeddings
        lastRoundEmbeds = embeddings[self.labelledIdsOverAL[-1]]
        dist = pairwise_distances(embeddings, lastRoundEmbeds, metric="euclidean")
        minDist = np.min(dist, axis=1).reshape(-1, 1)

        # create new query batch one by one
        newBatch = []
        nCycled = []
        while len(newBatch) < self.argsForThisRun['n_queries_per_al_iteration']:
            ind = np.argmax(minDist)
            assert ind not in self.labelledIdsOverAL[-1] and ind not in newBatch
            lastRoundEmbeds = embeddings[[ind]]
            dist = pairwise_distances(embeddings, lastRoundEmbeds, metric="euclidean")
            minDist = np.minimum(minDist, dist)

            # for this specific strategy only, if the model is confident about the current id
            # as bool'd by the mask ge to conf threshold, then don't append it to the batch but continue searching
            if self.queryStrategy == 'kcenter_hybrid' and masks[ind] == 1:
                continue

            # if rejection sampling is being used, similar to above only append if target different from pseudolabel
            nCycled.append(ind)
            if 'use_rejection_sampling' in self.argsForThisRun and self.argsForThisRun['use_rejection_sampling']:
                if pseudo_labels[ind] != targets[ind]:
                    newBatch.append(ind)
                else:
                    continue
            else:
                newBatch.append(ind)

        self.nCycledThrough.append(nCycled)
        del lastTTobj, minDist, dist, lastRoundEmbeds, embeddings

        return newBatch

    def random_with_rejection_sampling(self):

        lastTTobj = self.trainTracksOverAL[-1]
        original_model, _ = load_model(lastTTobj.args)

        if self.trainTracksOverAL[-1].args['ema']:
            lastEMAmod = EMA(original_model, 0.999)
            original_model.load_state_dict(lastTTobj.model)
            lastEMAmod.load_state_dict(lastTTobj.ema_model)
            lastEMAmod.apply_shadow()
        else:
            original_model.load_state_dict(lastTTobj.model)

        try:
            p_data = lastTTobj.p_data
            p_model = lastTTobj.p_model
        except:
            print('AL loop: No p_data or p_model recorded, setting up new bland objects...')
            p_data = torch.as_tensor(np.repeat(1 / 20, 20), device=self.argsForThisRun['device'])
            p_model = PMovingAverage(nclass=20, buf_size=128, device=self.argsForThisRun['device'])

        ids, pseudo_labels, targets, masks, scores, _ = get_pseudo_labels_and_masks(original_model,
                                                                                 p_data=p_data,
                                                                                 p_model=p_model,
                                                                                 args=self.argsForThisRun)

        latestLabelledIds = np.unique(sorted(self.labelledIdsOverAL[-1]))
        possible_labeled_idx = np.array(np.arange(self.allIdsLen))
        assert np.all(ids.tolist() == possible_labeled_idx.tolist()), 'Ids should be the same for next bit to work'
        subsetPreviousids = possible_labeled_idx[latestLabelledIds]
        assert np.all(subsetPreviousids == latestLabelledIds)
        subsetPreviousRemoved = np.array(np.setdiff1d(np.array(possible_labeled_idx), np.array(subsetPreviousids)))

        # here the vector of targets is acting as the labelling human
        # whereas if it was for real, this would be replaced with the label the human had just assigned
        newBatch = []
        nCycled = []
        while len(newBatch) < self.argsForThisRun['n_queries_per_al_iteration']:
            # update the subset of ids not to choose from because already chosen
            # and choose one id at a time to compare the pseudo label to real target
            subsetPreviousRemoved = np.array(np.setdiff1d(subsetPreviousRemoved, np.array(newBatch)))
            potentialAddition = self.rndGen.choice(subsetPreviousRemoved, 1, replace=False)
            nCycled.append(potentialAddition[0])
            if pseudo_labels[potentialAddition] != targets[potentialAddition]:
                newBatch.append(potentialAddition[0])

        self.nCycledThrough.append(nCycled)
        return newBatch

    def custom_querying(self):

        lastTTobj = self.trainTracksOverAL[-1]
        mod, _ = load_model(lastTTobj.args)

        if self.trainTracksOverAL[-1].args['ema']:
            lastEMAmod = EMA(mod, 0.999)
            mod.load_state_dict(lastTTobj.model)
            lastEMAmod.load_state_dict(lastTTobj.ema_model)
            lastEMAmod.apply_shadow()
        else:
            mod.load_state_dict(lastTTobj.model)

        # get the embeddings and corresponding ids for all the training data
        ids1, original_embeddings = get_embeddings_for_querying(mod, self.argsForThisRun)

        # pdata and pmodel
        try:
            p_data = lastTTobj.p_data
            p_model = lastTTobj.p_model
        except:
            print('AL loop: No p_data or p_model recorded, setting up new bland objects...')
            p_data = torch.as_tensor(np.repeat(1 / 20, 20), device=self.argsForThisRun['device'])
            p_model = PMovingAverage(nclass=20, buf_size=128, device=self.argsForThisRun['device'])

        # get the by-class query budget using p_data and the query budget in args
        budget = self.argsForThisRun['n_queries_per_al_iteration']
        per_class_probs = (1 - copy.deepcopy(p_data).cpu().numpy()) / (1 - copy.deepcopy(p_data).cpu().numpy()).sum()
        byClassBudget = np.floor(per_class_probs * budget)
        byClassBudget[byClassBudget == 0] = 1
        left_in_budget = budget - byClassBudget.sum()
        if left_in_budget > 0:
            should_have_rounded_up = np.argsort((per_class_probs * budget) - np.floor(per_class_probs * budget))[
                                     ::-1]
            for i in should_have_rounded_up:
                byClassBudget[i] = byClassBudget[i] + 1
                left_in_budget -= 1
                if left_in_budget == 0:
                    assert byClassBudget.sum() == budget
                    byClassBudget = byClassBudget.astype(int)
                    break

        # get pseudo labels
        # again obviously shouldn't be using targets unless where could be replaced by human
        ids, original_pseudo_labels, original_targets, original_masks, original_scores, _ = get_pseudo_labels_and_masks(original_model=mod,
                                                                                 p_data=p_data,
                                                                                 p_model=p_model,
                                                                                 args=self.argsForThisRun)
        # ids; make sure everything is aligned
        setLen = len(ids)
        possible_labeled_idx = np.array(np.arange(setLen))

        subsetPreviousids = possible_labeled_idx[np.unique(sorted(self.labelledIdsOverAL[-1]))]
        #check that target balance is the same as when loaded
        print('AL loop: previous label distribution was:', np.unique(original_targets[subsetPreviousids],
                                                                     return_counts=True),
              np.bincount(original_targets[subsetPreviousids]).sum())



        # for each class select the pseudo labels that match, select the highest confidence ones
        # get the ordered most similar as in kcenter
        # mask out the ones that are also above the confidence threshold
        # so choose the smallest index which is under the confidence threshold
        # check with rejection sampling; add to queries if labels are a surprise, don't add if not
        # continue while the budget for that class is not met
        newBatch = []
        nCycled = []
        for i in range(len(p_data)):

            # subset everything by the ids because these get used in their own right in the next part
            # so it's not possible to just let the ids do all the work
            # do this in the loop so that there is a reset when we get to the next class
            subsetPreviousRemoved = np.array(np.setdiff1d(np.array(possible_labeled_idx), np.array(subsetPreviousids)))
            pseudo_labels = original_pseudo_labels[subsetPreviousRemoved]
            targets = original_targets[subsetPreviousRemoved]
            masks = original_masks[subsetPreviousRemoved]
            scores = original_scores[subsetPreviousRemoved]
            embeddings = original_embeddings[subsetPreviousRemoved, :]

            # find the indices which match that class label
            # select most confident pseudo-labels
            byClassPseudoLabInds = np.where(pseudo_labels == i)[0]

            if len(byClassPseudoLabInds) == 0:
                print('Al loop: There arent any pseudolabels to pick from for class', i)
                print('choosing from examples which have already been labelled instead...')
                pseudo_labels = original_pseudo_labels
                targets = original_targets[subsetPreviousids]  # i.e. only the labels we do have at this stage
                masks = original_masks
                scores = original_scores
                embeddings = original_embeddings
                subsetPreviousRemoved = ids
                # now can get the inds where we already have targets
                byClassPseudoLabInds = np.where(targets == i)[0]  # though these are actual labels now

            byClassPseudoLabScoreMaxInd = np.argmax(scores[byClassPseudoLabInds])
            originalIndForPseudoLabMax = byClassPseudoLabInds[byClassPseudoLabScoreMaxInd]
            #assert pseudo_labels[originalIndForPseudoLabMax] == i, 'Pseudo label doesnt match up'
            orderedMostSimilar = np.argsort(pairwise_distances(X=embeddings[originalIndForPseudoLabMax,:].reshape(1, -1),
                                                               Y=embeddings,
                                                               metric="euclidean"))
            reverseMask = np.where(masks == 0., True, False)  # this means we only look at under conf threshold
            reorderedReverseMask = reverseMask[orderedMostSimilar]
            # the next line re-aligns the ids uncovered with the ids of the subsetPreviousRemoved vector
            orderedMaskedMostSimilar = subsetPreviousRemoved[orderedMostSimilar[reorderedReverseMask]].tolist()
            [orderedMaskedMostSimilar.remove(i) for i in orderedMaskedMostSimilar if i in subsetPreviousids]
            classBatch = []
            j = 0  # use this to traverse the list of potential ids
            while len(classBatch) < byClassBudget[i]:
                # update the subset of ids not to choose from because already chosen
                # and choose one id at a time to compare the pseudo label to real target
                runningRemoved = subsetPreviousids.tolist() + classBatch + newBatch
                if orderedMaskedMostSimilar[j] in runningRemoved:
                    j += 1
                    continue
                assert orderedMaskedMostSimilar[j] in subsetPreviousRemoved, "Have a look here for the problem"
                try:
                    potentialAddition = orderedMaskedMostSimilar[j]
                except:
                    # if the above fails because the index has gone beyond the length of the array
                    # then the method has basically expired
                    # so use random sampling instead
                    potentialAddition = self.rndGen.choice(subsetPreviousRemoved, 1, replace=False)
                # insert condition for rejection sampling
                nCycled.append(potentialAddition)
                if self.argsForThisRun['use_rejection_sampling']:
                    if original_pseudo_labels[potentialAddition] != original_targets[potentialAddition]:
                        classBatch.append(potentialAddition)
                    else:
                        j += 1
                        continue
                else:
                    classBatch.append(potentialAddition)
                j += 1

            newBatch.append(classBatch)
            newBatch = np.concatenate(np.asarray(newBatch, dtype=object), axis=None).tolist()

        self.nCycledThrough.append(nCycled)
        return newBatch

    def least_certain(self):

        lastTTobj = self.trainTracksOverAL[-1]
        mod, _ = load_model(lastTTobj.args)

        if self.trainTracksOverAL[-1].args['ema']:
            lastEMAmod = EMA(mod, 0.999)
            mod.load_state_dict(lastTTobj.model)
            lastEMAmod.load_state_dict(lastTTobj.ema_model)
            lastEMAmod.apply_shadow()
        else:
            mod.load_state_dict(lastTTobj.model)

        # pdata and pmodel
        try:
            p_data = lastTTobj.p_data
            p_model = lastTTobj.p_model
        except:
            print('AL loop: No p_data or p_model recorded, setting up new bland objects...')
            p_data = torch.as_tensor(np.repeat(1 / 20, 20), device=self.argsForThisRun['device'])
            p_model = PMovingAverage(nclass=20, buf_size=128, device=self.argsForThisRun['device'])

        ids, pseudo_labels, targets, masks, scores, outputs = get_pseudo_labels_and_masks(original_model=mod,
                                                                                 p_data=p_data,
                                                                                 p_model=p_model,
                                                                                 args=self.argsForThisRun)

        # code to retrieve the max entropies was taken from
        # https://github.com/microsoft/CameraTraps/blob/
        # f69de8339bf19806fbe943963e2b385b948c35b2/research/active_learning/
        # active_learning_methods/entropy_sampling.py
        # code to retrieve the margins was taken from
        # https://github.com/microsoft/CameraTraps/blob/
        # f69de8339bf19806fbe943963e2b385b948c35b2/research/
        # active_learning/active_learning_methods/margin_AL.py
        if self.queryStrategy == 'least_certain':
            entropies = np.apply_along_axis(stats.entropy, 1, outputs)
            sortedIds = np.argsort(entropies)[::-1]
        elif self.queryStrategy == 'margin':
            sort_distances = np.sort(outputs, 1)[:, -2:]
            min_margin = sort_distances[:, 1] - sort_distances[:, 0]
            sortedIds = np.argsort(min_margin)
        else:
            raise Exception('There isnt a suitable query strat')

        newBatch = []
        nCycled = []
        i = 0
        while len(newBatch) < self.nQueriesPerALIter:
            if sortedIds[i] not in self.labelledIdsOverAL[-1]:
                nCycled.append(sortedIds[i])
                if self.argsForThisRun['use_rejection_sampling']:
                    if pseudo_labels[sortedIds[i]] != targets[sortedIds[i]]:
                        newBatch.append(sortedIds[i])
                    else:
                        i += 1
                        continue
                else:
                    newBatch.append(sortedIds[i])
            i += 1

        self.nCycledThrough.append(nCycled)
        return newBatch

    def choose_from_predetermined_random(self, filepath_1600):
        with open(filepath_1600, 'rb') as handle:
            poss_1600 = pickle.load(handle)
        handle.close()
        latestLabelledIds = np.unique(sorted(self.labelledIdsOverAL[-1]))
        subsetPreviousRemoved = np.array(np.setdiff1d(np.array(poss_1600), np.array(latestLabelledIds)))
        if len(subsetPreviousRemoved) != 0:
            # this is the case throughout the whole loop
            return self.rndGen.choice(subsetPreviousRemoved, self.nQueriesPerALIter, replace=False).tolist()
        else:
            print('AL loop: This is the last loop and the queries are finished up; '
                  'returning an empty list as it doesnt matter now')
            return []

    def create_AL_checkpoint(self):
        torch.save(self, self.al_tracking_path)
        print('AL loop: Active Learning Checkpoint Created')


def get_embeddings_for_querying(original_model, args):
    modeldc = copy.deepcopy(original_model)
    modeldc.fc = nn.Identity()

    _, _, test_transform = im_transforms(args)
    whole_train_set = images_data(args, ['train'], test_transform, True)
    # really important that shuffle=False here
    wts = DataLoader(whole_train_set, batch_size=args['batch_size'], shuffle=False,
                     num_workers=args['num_workers'], pin_memory=True)

    with torch.no_grad():
        im_ids = []
        embeds = []

        for i, data in enumerate(wts, 0):
            inputs = data['im']
            context_sheet_ids = data['original_id']

            inputs = inputs.to(args["device"])
            outputs = modeldc(inputs)

            im_ids.append(context_sheet_ids)
            embeds.append(outputs.detach().cpu().numpy())

        ids = np.concatenate(im_ids, axis=None)
        embeddings = np.concatenate(embeds, axis=0)

    # this test is just here to check the ordering of the ids has come out
    # the same as when it went in. Then can safely use the same external index
    # on the output of this as would have anyway (e.g. on random)
    assert (whole_train_set.original_ids.tolist() == ids.tolist()), 'These ids and embeddings are going to need sorting'

    indexedIds = np.array(np.arange(len(ids)))

    del _, test_transform, whole_train_set, wts, modeldc, inputs, outputs

    return indexedIds, embeddings


def get_pseudo_labels_and_masks(original_model, p_data, p_model, args):
    modeldc = copy.deepcopy(original_model)

    weak_transform, _, _ = im_transforms(args)
    whole_train_set = images_data(args, ['train'], weak_transform, True)
    # really important that shuffle=False here
    wts = DataLoader(whole_train_set, batch_size=args['batch_size'] * args['fm_ratio_mu'], shuffle=False,
                     num_workers=args['num_workers'], pin_memory=True)

    ids, pseudo_labels, targets, masks, scores_all, outputs = retrieve_fixmatch_outputs(modeldc, wts, args, p_data, p_model)

    assert np.all(
        whole_train_set.original_ids.tolist() == ids.tolist()), 'These ids and embeddings are going to need sorting'

    indexedIds = np.array(np.arange(len(ids)))

    return indexedIds, pseudo_labels, targets, masks, scores_all, outputs
