##################################################
# Purpose: Train loop for sup/FM/eval            #
# Author: Amy Andrews                            #
# Resources used:
# Pytorch documentation https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# Pytorch documentation https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
##################################################
import json

import copy
import torch
import numpy as np
import time
from fixmatch import fixmatch
from utils import *
from ema import EMA
from torch.utils.data import DataLoader
from pprint import pprint

from query_strat import active_learning_querying
from load_data import im_transforms, images_data

import config

# train the model using a backbone initialised model and other model components
def train_model(model, optimiser, scheduler, train, val, args, checkpoint=None):
    """

    :param model: the initialised model type, torch model
    :param optimiser: optimiser, torch object
    :param scheduler: learning rate scheduler, torch object
    :param train: training set or labelled/unlabelled training sets, dataloader or dataloader tuple
    :param val: validation set, dataloader
    :param args: args dictionary from main, dict
    :param checkpoint: checkpoint if given
    :return: model, results dictionary
    """

    start = time.time()
    torch.manual_seed(args['seed'])
    rnd = np.random.default_rng(seed=args['seed'])
    max_epochs = args['epochs']
    if args['ema']:
        ema_model = EMA(model, 0.999)
    else:
        ema_model = None

    # if there's a checkpoint set up for this then use it
    if checkpoint is not None:
        track = checkpoint
        optimiser.load_state_dict(checkpoint.optimiser)
        scheduler.load_state_dict(checkpoint.scheduler)
        model.load_state_dict(checkpoint.model)
        if checkpoint.ema_model is not None:
            ema_model.load_state_dict(checkpoint.ema_model)
        track.args = args
        print('Updating arguments to match current set args so that args are now:')
        pprint(track.args)
        print('Loading pretrained', args['pretrained_model'])
        print(f'Previously finished at epoch {checkpoint.epoch} with loss {checkpoint.epoch_loss}\n'
              f'Restarting from epoch {checkpoint.epoch + 1}')
        print('\n**********************************')

    else:
        track = TrainTrack(name=args['train_loss'], model=model, ema_model=ema_model,
                           optimiser=optimiser, scheduler=scheduler, args=args)

    # if this is for AL then set up tracking
    # this is the first iteration so pick the next queries here
    if args['using_al']:
        weak_transform, strong_transform, test_transform = im_transforms(args)
        whole_train_set = images_data(args, ['train'], weak_transform, True)
        al_tracker = active_learning_querying(trainTrackObj=copy.deepcopy(track),
                                              allIdsLen=len(whole_train_set),
                                              queryStrategy=args['query_strat'],
                                              nQueriesPerIter=args['n_queries_per_al_iteration'],
                                              argsForThisRun=args)
        if args['pretrained_model'] != '' and track.epoch == 132:
            print('Picking up from a pretrained model so getting new queries first...')
            expandedTotalIdsForNextIter = al_tracker.query_selection()
            supset = images_data(args, split=['train'], transform=weak_transform, return_single_image=True,
                                         indices=expandedTotalIdsForNextIter)
            train[0] = DataLoader(supset, batch_size=args['batch_size'], shuffle=True,
                                    num_workers=args['num_workers'],
                                    drop_last=True, pin_memory=True)
            print('\n***Now have', len(al_tracker.labelledIdsOverAL[-1]), 'labels available***\n')
        else:
            # start from the beginning
            print('Starting afresh so no queries right now, just set up tracker...')

    # prepare the data
    if isinstance(train, list):  # which means FM really
        supDataloaderIterator = iter(train[0])
        unsupDataloaderIterator = iter(train[1])
        if args['use_da']:
            if hasattr(track, 'p_data') and track.p_data is not None:
                p_data = track.p_data
            else:
                if args['labelling_split_type'] == 'true_uniform_random':
                    f = open('splits/ss_animals_capture.json')
                    ss_counts = json.load(f)
                    p_data = np.asarray(ss_counts) / np.sum(np.asarray(ss_counts))
                    f.close()
                else:
                    p_data = np.bincount(train[0].dataset.targets) / len(train[0].dataset)
                assert len(p_data) == train[0].dataset.num_classes
                p_data = torch.as_tensor(p_data, device=args['device'])
                # print('prob of data:', p_data)
            if hasattr(track, 'p_model') and track.p_model is not None:
                p_model = track.p_model
            else:
                p_model = PMovingAverage(nclass=train[0].dataset.num_classes, buf_size=128, device=args['device'])
        else:
            p_data = None
            p_model = None

    else:
        trainDataloaderIterator = iter(train)
        p_data = None
        p_model = None

    while track.epoch < max_epochs:
        complete_epoch_time_start = time.time()
        track.epoch += 1
        print('\nEpoch {}/{}'.format(track.epoch, max_epochs), '\n', ('-' * 10))

        for phase in ['train', 'test']:
            ########################################################################################################
            if phase == 'train':
                train_start = time.time()
                model.train()
                if args['train_loss'] == 'supervised':
                    trainDataloaderIterator, optimiser, scheduler, \
                    model, ema_model, track = sup_train(ds=train,
                                                        trainDataloaderIterator=trainDataloaderIterator,
                                                        optimiser=optimiser,
                                                        scheduler=scheduler, model=model,
                                                        args=args, ema_model=ema_model,
                                                        track=track)
                elif args['train_loss'] == 'fixmatch':

                    # execute fixmatch training procedure
                    supDataloaderIterator, unsupDataloaderIterator, \
                    optimiser, scheduler, model, \
                    ema_model, p_model, track = fixmatch(train=train,
                                                         supDataloaderIterator=supDataloaderIterator,
                                                         unsupDataloaderIterator=unsupDataloaderIterator,
                                                         model=model,
                                                         confidence_threshold=args['fm_conf_threshold'],
                                                         unlabelled_loss_weight=args['fm_loss_weight_lmda'],
                                                         args=args,
                                                         optimiser=optimiser,
                                                         scheduler=scheduler,
                                                         ema_model=ema_model, p_data=p_data, p_model=p_model,
                                                         track=track)

                    # again if the AL loop then add the model that was trained on the above ids
                    # at its final epoch, i.e. the epoch before the model is updated with new supervised ids
                    if args['using_al'] and \
                            track.epoch in np.arange(132, args['epochs'] + 1, args['al_iteration_every_n_epochs']):
                        # save the previous model as the one which was trained on the previous queries
                        al_tracker.add_new_traintrack(trainTrackObj=copy.deepcopy(track))
                        # get queries and load the data for them
                        # this is ready for the next epoch
                        expandedTotalIdsForNextIter = al_tracker.query_selection()
                        supset = images_data(args, split=['train'], transform=weak_transform, return_single_image=True,
                                             indices=expandedTotalIdsForNextIter)
                        train[0] = DataLoader(supset, batch_size=args['batch_size'], shuffle=True,
                                              num_workers=args['num_workers'],
                                              drop_last=True, pin_memory=True)
                        print('\n***Now have', len(al_tracker.labelledIdsOverAL[-1]), 'labels available***\n')
                        supDataloaderIterator = iter(train[0])
                        if args['use_da']:
                            if args['labelling_split_type'] == 'true_uniform_random':
                                f = open(animal_dicts)
                                ss_counts = json.load(f)
                                p_data = np.asarray(ss_counts) / np.sum(np.asarray(ss_counts))
                                f.close()
                            else:
                                p_data = np.bincount(train[0].dataset.targets) / len(train[0].dataset)
                            assert len(p_data) == train[0].dataset.num_classes
                            p_data = torch.as_tensor(p_data, device=args['device'])
                        al_tracker.create_AL_checkpoint()


                ########################################################################################################
                if args['scheduler'] == 'plateau_train':
                    scheduler.step(track.epoch_loss)  # this monitors train loss plateauing
                time_elapsed = time.time() - train_start
                print('Training phase complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
            ########################################################################################################
            else:  # i.e. test (validation) phase
                test_start = time.time()
                model.eval()
                epoch_loss, epoch_acc = val_step(val, model, args, ema_model)
                track.update(model, ema_model, optimiser, scheduler, sup_loss=0,
                             unsup_loss=0, epoch_loss=epoch_loss, epoch_acc=epoch_acc, phase=phase,
                             p_model=p_model, p_data=p_data, using_val=args['using_val_to_train'])
                # having the scheduler as 'plateau' necessitates a validation set,
                # so use this validation set to track the loss, otherwise use train
                if args['scheduler'] == 'plateau':
                    assert args['using_val_to_train'], 'Set using_val_to_train to True first'
                    scheduler.step(track.epoch_loss)  # this monitors validation loss plateauing
                time_elapsed = time.time() - test_start
                print('Validation phase complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
            ########################################################################################################
        track.create_checkpoint()
        if args['using_val_to_train'] and track.patience >= args['patience']:
            print("Stopping early at epoch: ", track.epoch)
            break
        final_time = time.time() - complete_epoch_time_start
        print('One complete epoch done in: {:.0f}m {:.4f}s'.format(final_time // 60, final_time % 60))
    # end of epoch loop

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, ema_model, track.output_results()
