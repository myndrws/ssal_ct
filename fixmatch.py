##################################################
# Purpose: Run FixMatch procedure given inputs   #
# Author: Amy Andrews                            #
# Resources used:
# Official FixMatch implementation at
# https://github.com/google-research/fixmatch/tree/d4985a158065947dba803e626ee9a6721709c570
# and unofficial PyTorch FixMatch implementation at
# https://github.com/kekmodel/FixMatch-pytorch/tree/ced359021e0b48bf3be48504f98ea483579c6e71/models
# in addition to https://github.com/Celiali/FixMatch
##################################################

import numpy as np
import torch
import torch.nn.functional as F
import time
from utils import AverageMeter

def fixmatch(train, supDataloaderIterator, unsupDataloaderIterator, model, confidence_threshold,
             unlabelled_loss_weight, args, optimiser, scheduler, ema_model, p_data, p_model, track):

    """
    This module is intended to instantiate the FixMatch algorithm for a single epoch
    This code is explicitly based on the theoretical work of the authors of FixMatch,
    from Sohn et al, 2020, available at arXiv:2001.07685
    as well as the practical work available at https://github.com/google-research/fixmatch

    :param track:
    :param scheduler:
    :param ema_model:
    :param train: tuple of supervised and unsupervised dataloader objects
    :param train: the train tuple to recreate the iterables from
    :param model:
    :param confidence_threshold:
    :param unlabelled_ratio_mu:
    :param unlabelled_loss_weight:
    :param args: the args for the experiment
    :param optimiser: the optimiser to pass in and out
    :return:
    """

    running_loss = AverageMeter()
    loss_s = AverageMeter()
    loss_u = AverageMeter()
    av_time = AverageMeter()
    sup_acc = AverageMeter()
    pseudo_u_acc = AverageMeter()
    av_above_threshold = AverageMeter()
    time_to_load_batches = []
    time_to_gather_inputs = []
    time_for_model_inputs = []
    time_for_backprop = []
    time_to_update_trackers =[]

    for batch_idx in range(args['n_iters_per_epoch']):

        start = time.time()
        # loop through the data and reset the iter objects
        # if there are more iters than length of the objects
        # (which there will be by default)
        # as found on this pytorch issue https://github.com/pytorch/pytorch/issues/1917

        start_of_time_to_load_batches = time.time()
        try:
            s_batch = next(supDataloaderIterator)
        except StopIteration:
            supDataloaderIterator = iter(train[0])
            s_batch = next(supDataloaderIterator)
        try:
            u_batch = next(unsupDataloaderIterator)
        except StopIteration:
            unsupDataloaderIterator = iter(train[1])
            u_batch = next(unsupDataloaderIterator)
        time_to_load_batches.append(time.time() - start_of_time_to_load_batches)

        start_of_gathering_inputs = time.time()

        batch_size = s_batch['im'].shape[0]
        inputs_s = s_batch['im']
        inputs_u_weak = u_batch['weak_augmentation']
        inputs_u_strong = u_batch['strong_augmentation']
        labels_u = u_batch['target']
        labels_s = s_batch['target']

        inputs = torch.cat((inputs_s,  # this is also a weak augmentation
                            inputs_u_weak,
                            inputs_u_strong)).to(args['device'])
        labels_s = labels_s.to(args['device'])
        labels_u = labels_u.to(args['device'])

        #print(labels_s.get_device())

        optimiser.zero_grad(set_to_none=True)
        #print(torch.cuda.memory_summary(args['device']))
        time_to_gather_inputs.append(time.time() - start_of_gathering_inputs)

        # forward
        start_of_model_inputs = time.time()
        with torch.set_grad_enabled(True):
            logits = model(inputs)
            logits_s = logits[:batch_size]
            logits_u_weak, logits_u_strong = logits[batch_size:].chunk(2)
            del logits

            Ls = F.cross_entropy(logits_s, labels_s, reduction='mean')
            olwp = torch.softmax(logits_u_weak.detach(), dim=-1)

            #da
            if args['use_da']:
                # update model probabilities and ratio of data probability to model probability
                p_model_y = sum(olwp) / len(logits_u_weak)
                p_model_y = torch.tile(p_model_y, [len(logits_u_weak), 1])

                # use this to alter the output probabilities for the unlabelled weak augmentation inputs
                # this will be used for creating the pseudo label instead of original outputs
                #print('pretransformed olwp:', olwp)
                p_ratio = (1e-6 + p_data) / (1e-6 + p_model())
                outputs_labelled_weak_prob = olwp * p_ratio
                outputs_labelled_weak_prob /= torch.sum(outputs_labelled_weak_prob, dim=1, keepdim=True)
                #print('outputs matrix should be 448*20:', outputs_labelled_weak_prob, outputs_labelled_weak_prob.size())
                # update the probability of unlabelled guessing
                p_model.update(p_model_y)
                del p_ratio, p_model_y
            else:
                outputs_labelled_weak_prob = olwp

            scores, pseudo_label = torch.max(outputs_labelled_weak_prob, dim=-1)
            mask = scores.ge(confidence_threshold).float()
            Lu = (F.cross_entropy(logits_u_strong, pseudo_label, reduction='none') * mask).mean()

            loss = Ls + unlabelled_loss_weight * Lu

            time_for_model_inputs.append(time.time() - start_of_model_inputs)

            start_of_time_for_backprop = time.time()
            # this function is only for training
            # so always do loss backward and optimiser step
            loss.backward()
            optimiser.step()  # scheduler step done in train_loop
            if args['scheduler'] == 'cosine':
                scheduler.step()
            if args['ema']:
                ema_model.update_params()

        if args['ema']:
            ema_model.update_buffer()

        time_for_backprop.append(time.time() - start_of_time_for_backprop)

        start_of_time_for_updating_trackers = time.time()

        running_loss.update(loss.item())
        loss_s.update(Ls.item())
        loss_u.update(Lu.item())

        _, s_preds = torch.max(logits_s, 1)

        sup_acc.update((((s_preds == labels_s).sum())/len(s_preds)).item())
        pseudo_u_acc.update(((((pseudo_label == labels_u) * mask).sum())/len(mask)).item())
        av_above_threshold.update((mask.sum()/u_batch['weak_augmentation'].shape[0]).item())

        time_to_update_trackers.append(time.time() - start_of_time_for_updating_trackers)

        del mask, s_preds, labels_s, labels_u, inputs_u_weak, inputs_u_strong, inputs_s, inputs
        del logits_s, logits_u_weak, logits_u_strong, scores, pseudo_label
        del s_batch, u_batch, olwp, outputs_labelled_weak_prob

        av_time.update((time.time() - start))

    # end of iteration through dataloader batches
    print("Finished epoch", '------')
    print('Average time for batch loading:', np.mean(time_to_load_batches))
    print('Average time for gathering inputs:', np.mean(time_to_gather_inputs))
    print('Average time for putting inputs through model:', np.mean(time_for_model_inputs))
    print('Average time for backprop:', np.mean(time_for_backprop))
    print('Average time for updating trackers:', np.mean(time_to_update_trackers))
    print('Average iteration time is {:.0f}m {:.4f}s'.format(av_time.avg // 60, av_time.avg % 60))
    print('Supervised loss:', loss_s.avg, '--- Unsupervised loss:', loss_u.avg)
    print('Average unlabelled above confidence threshold:', av_above_threshold.avg)
    print('Supervised accuracy:', sup_acc.avg,
          '--- True pseudo-label accuracy:', pseudo_u_acc.avg)
    track.update(model, ema_model, optimiser, scheduler, loss_s.avg,
                 loss_u.avg, epoch_loss=running_loss.avg, epoch_acc=sup_acc.avg,
                 phase='train', p_model=p_model, p_data=p_data, using_val=args['using_val_to_train'])

    del time_for_backprop, time_for_model_inputs, time_to_gather_inputs, time_to_load_batches, \
        time_to_update_trackers, start, start_of_model_inputs, start_of_gathering_inputs, \
        start_of_time_to_load_batches, start_of_time_for_backprop, start_of_time_for_updating_trackers
    del running_loss, loss_u, loss_s, loss

    return supDataloaderIterator, unsupDataloaderIterator, optimiser, scheduler, model, ema_model, p_model, track
