##################################################
# Purpose: Utilities and useful functions        #
# Author: Amy Andrews                            #
# Resources used:
# https://github.com/google-research/fixmatch/blob/d4985a158065947dba803e626ee9a6721709c570/libml/layers.py#L125
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://github.com/Celiali/FixMatch/blob/main/experiments/experiment.py
# Matplotlib documentation https://matplotlib.org/
# scikit learn https://scikit-learn.org/stable/index.html
##################################################

from collections import Counter
import time
import torch.nn as nn
import math
import copy
import datetime as dt
import pickle

import numpy as np
import torch
import json
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from torchvision import models, transforms
import matplotlib.image as mpimg
from torchvision.utils import make_grid
from sklearn.metrics import confusion_matrix, classification_report, f1_score, top_k_accuracy_score

from ema import EMA

import config

def counting(to_count):
    n_unique = len(np.unique(to_count))
    print(f'Total Classes:{n_unique}')
    if n_unique < 50:
        print('Printing all classes\n')
        count = Counter(to_count).most_common(n_unique)
        for val in count:
            print(val[0], ' : ', val[1])
    else:
        print('Printing top 20 most common classes\n')
        count = Counter(to_count).most_common(20)
        for val in count:
            print(val[0], ' : ', val[1])


def load_model(args, load_from_path='', device=None):
    """
    :param args: the arguments used to load the model
    :param load_from_path: path to load the model from if required
    :param device: device to load model on if required
    :return: the appropriate model defined by args - pretrained with imnet or not, model type, checkpointed or not
    """

    if device==None:
        device=args['device']

    checkpoint=None

    if args['mod_type'] == 'rn18':

        if args['imnet_pretrained']:
            model = models.resnet18(pretrained=True)
        else:
            model = models.resnet18(pretrained=False)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args['projection_dims'])
        model = model.to(device)

    elif args['mod_type'] == 'wrn28':
        import wideresnet as wrn
        model = wrn.build_wideresnet(depth=28,
                                     widen_factor=2,
                                     dropout=0,
                                     num_classes=args['projection_dims'])
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args['projection_dims'])
        model = model.to(device)

    if load_from_path != '':
        checkpoint = torch.load(load_from_path)
        model.load_state_dict(checkpoint.model)
        model.to(device)
        print(f'loading trained model from best epoch {checkpoint.epoch}, '
              f'which had at that epoch a loss of {checkpoint.epoch_loss}')

    return model, checkpoint



def eval_on_test(test, model, device, embeddings_only=False):
    model.eval()
    with torch.no_grad():

        preds = []
        targets = []
        im_ids = []
        outs = []
        cs_ids = []
        im_paths = []

        for i, data in enumerate(test, 0):

            inputs = data['im']
            indx = data['id']
            labels = data['target']
            context_sheet_ids = data['original_id']
            im_path = data['img1_path']

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            target_labels = labels.detach().cpu().numpy()
            targets.append(target_labels)
            images_ids = indx.detach().cpu().numpy()
            im_ids.append(images_ids)
            cs_ids.append(context_sheet_ids)
            im_paths.append(im_path)

            # add clause for returning predictions/outputs by default
            if not embeddings_only:
                sm = nn.Softmax(dim=1)
                _, p = torch.max(outputs, 1)
                predictions = p.detach().cpu().numpy()
                preds.append(predictions)
                outputs_over_classes = sm(outputs).detach().cpu().numpy()
                outs.append(outputs_over_classes)
            else:
                outs.append(outputs.detach().cpu().numpy())

        res = {}

        res['im_ids'] = np.concatenate(im_ids, axis=None)
        res['targets'] = np.concatenate(targets, axis=None)
        res['outputs'] = np.concatenate(outs, axis=0)
        res['context_sheet_ids'] = np.concatenate(cs_ids, axis=None)
        res['im_paths'] = np.concatenate(im_paths, axis=None)

        # if it's not embeddings, so outputs are torch max, return this in dict
        if not embeddings_only:
            res['preds'] = np.concatenate(preds, axis=None)

    return res


# determine the original size of the images provided
def av_im_size(im_path_list, args):
    height = []
    width = []
    for i in range(len(im_path_list)):
        img = mpimg.imread(args['data_dir'] + im_path_list[i])
        height.append(img.shape[0])
        width.append(img.shape[1])
    return height, width

# load images in a less memory intensive way
def load_all_data_in_memory(dataloader):
    images = []
    imageids = []
    targets = []

    for i, data in enumerate(dataloader, 0):
        images.extend(data['im'])
        imageids.extend(data['original_id'])
        targets.extend(data['target'])

    dictionary = {'imageids': imageids, 'images': images, 'targets': targets}

    return dictionary


# load embeddings from a model
def retrieve_embeddings(model, inputs, device):
    """
    :param model: the model to use to generate the embeddings
    :param inputs: the data to embed
    :param device: the device that you want the model to be converted to
    :return: results dictionary of the eval_on_test function: mainly embeddings for the data in the same order given
    """
    model.fc = nn.Identity()

    model.to(device)

    res = eval_on_test(inputs, model, device, embeddings_only=True)

    return res


class TrainTrack(object):

    def __init__(self, name, model, ema_model, optimiser, scheduler, args):
        self.name = name
        self.args = args
        self.reset(model, ema_model, optimiser, scheduler)
        #self.update_best()

    def reset(self, model, ema_model, optimiser, scheduler):
        self.model = copy.deepcopy(model.state_dict())
        self.ema_model = copy.deepcopy(ema_model.shadow) if ema_model is not None else None
        self.optimiser = copy.deepcopy(optimiser.state_dict())
        self.scheduler = copy.deepcopy(scheduler.state_dict())
        self.sup_loss = np.inf
        self.unsup_loss = np.inf
        self.epoch_loss = np.inf
        self.sup_loss_over_epochs = []
        self.unsup_loss_over_epochs = []
        self.train_loss_over_epochs = []
        self.val_loss_over_epochs = []
        self.train_accuracy_over_epochs = []
        self.val_accuracy_over_epochs = []
        self.patience = 0
        self.epoch = 0
        self.p_model = None
        self.p_data = None

    def update(self, model, ema_model, optimiser, scheduler, sup_loss,
               unsup_loss, epoch_loss, epoch_acc, phase, p_model, p_data, using_val=False):
        self.epoch_loss = epoch_loss
        if phase == 'train':
            self.model = copy.deepcopy(model.state_dict())
            self.ema_model = copy.deepcopy(ema_model.shadow) if ema_model is not None else None
            self.optimiser = copy.deepcopy(optimiser.state_dict())
            self.scheduler = copy.deepcopy(scheduler.state_dict())
            self.sup_loss = sup_loss
            self.unsup_loss = unsup_loss
            self.p_model = p_model
            self.p_data = p_data
            self.train_loss_over_epochs.append(epoch_loss)
            self.train_accuracy_over_epochs.append(epoch_acc)
            self.sup_loss_over_epochs.append(sup_loss)
            self.unsup_loss_over_epochs.append(unsup_loss)

        if phase == 'test':
            self.val_loss_over_epochs.append(epoch_loss)
            self.val_accuracy_over_epochs.append(epoch_acc)

        print('{} Loss: {:.4f}'.format(phase, epoch_loss))

    def output_results(self):
        res = {'sup_loss_over_epochs': self.sup_loss_over_epochs,
               'unsup_loss_over_epochs': self.unsup_loss_over_epochs,
               'train_loss_over_epochs': self.train_loss_over_epochs,
               'val_loss_over_epochs': self.val_loss_over_epochs,
               'train_accuracy_over_epochs': self.train_accuracy_over_epochs,
               'val_accuracy_over_epochs': self.val_accuracy_over_epochs,
               'total_epochs_trained': self.epoch
               #'epochs_taken_to_best': self.best_epoch,
               }
        return res

    def create_checkpoint(self):
        torch.save(self, self.args['mod_path'])
        print('Checkpoint created at epoch', self.epoch)
        if self.args['using_al'] and \
                self.epoch == 132:
            cppath = self.args['mod_path']
            cppath = cppath.replace(".pt", "") + '__CPforALatEpoch' + str(self.epoch) + ".pt"
            torch.save(self, cppath)
            print('Separate checkpoint created for AL at epoch', self.epoch)


# setting path strings
def assign_paths(args):
    """
    Simply sets the names of the model files to reflect the args used
    :param args: the args passed to train the model
    :return: a string for the model path, encompassed in args
    """
    cur_time = dt.datetime.now().strftime("%Y_%m_%d__%H_%M")
    descriptorString = args['dataset'] + '_' + args['train_loss'] + '_' + \
                       args['mod_type'] + '_' + args['scheduler'] + '_'

    if args['imnet_pretrained']:
        descriptorString = descriptorString + 'imnet_pt_'
    if args['n_available_labels'] != '':
        descriptorString = descriptorString + str(args['n_available_labels']) + 'availStartLabs_'
    if args['labelling_split_type'] != '':
        descriptorString = descriptorString + str(args['labelling_split_type'])
        if args['split_id']:
            descriptorString = descriptorString + str(args['split_id'])
    if args['query_strat'] != '':
        descriptorString = descriptorString + '_' + args['query_strat'] + '_queryStrat'
    if args['use_rejection_sampling']:
        descriptorString = descriptorString + '_RejSampling_'
    if args['return_context']:
        descriptorString = descriptorString + '_context_'
    if args['pretrained_model'] != '':
        descriptorString = descriptorString + '_PTmodel_'
    if 'using_al' in args:
        if args['using_al']:
            descriptorString = descriptorString + '_ActiveLearningLoop_'

    descriptorString = descriptorString + '_totalIters' + str(args['n_total_iters']) + '_'
    args['mod_path'] = args['mod_dir'] + descriptorString + cur_time + '.pt'
    args['res_path'] = args['results_dir'] + descriptorString + cur_time + '.pickle'

    return args


# supervised module for supervised runs or for test
# runs when in supervised train loss or in any test phase
def sup_train(trainDataloaderIterator, ds, optimiser, scheduler, model, args, ema_model, track):

    epoch_loss = AverageMeter()
    epoch_acc = AverageMeter()
    obj_func = nn.CrossEntropyLoss()
    av_time = AverageMeter()

    # iterate over data
    for batch in range(args['n_iters_per_epoch']):

        start = time.time()

        try:
            data = next(trainDataloaderIterator)
        except StopIteration:
            trainDataloaderIterator = iter(ds)
            data = next(trainDataloaderIterator)

        inputs = data['im']
        labels = data['target']

        inputs = inputs.to(args['device'])
        labels = labels.to(args['device'])

        optimiser.zero_grad(set_to_none=True)

        # forward
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = obj_func(outputs, labels)

            # backward
            loss.backward()
            optimiser.step()
            if args['ema']:
                ema_model.update_params()
            if args['scheduler'] == 'cosine':
                scheduler.step()

        epoch_loss.update(loss.item())
        epoch_acc.update(torch.sum(preds == labels.data).item()/len(inputs))

        del inputs, labels, outputs, _, preds

        av_time.update((time.time() - start))



    if args['ema']:
        ema_model.update_buffer()

    print('Train set loss:', epoch_loss.avg, '------- Train set accuracy:', epoch_acc.avg)
    print('Average iteration time is {:.0f}m {:.4f}s'.format(av_time.avg // 60, av_time.avg % 60))
    track.update(model, ema_model, optimiser, scheduler, sup_loss=0,
                 unsup_loss=0, epoch_loss=epoch_loss.avg, epoch_acc=epoch_acc.avg, phase='train',
                 using_val=args['using_val_to_train'])

    return trainDataloaderIterator, optimiser, scheduler, model, ema_model, track


def val_step(ds, model, args, ema_model):

    epoch_loss = AverageMeter()
    epoch_acc = AverageMeter()
    obj_func = nn.CrossEntropyLoss()

    if args['ema']:
        ema_model.apply_shadow()

    # iterate over data
    for i, data in enumerate(ds, 0):
        inputs = data['im']
        labels = data['target']

        inputs = inputs.to(args['device'])
        labels = labels.to(args['device'])

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = obj_func(outputs, labels)

        epoch_loss.update(loss.item())
        epoch_acc.update(torch.sum(preds == labels.data).item()/len(inputs))

    if args['ema']:
        ema_model.restore()

    del inputs, labels, outputs, _, preds
    print('Validation set loss:', epoch_loss.avg, '------- Validation set accuracy:', epoch_acc.avg)

    return epoch_loss.avg, epoch_acc.avg


# from replication paper at https://github.com/Celiali/FixMatch/blob/main/experiments/experiment.py
def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    """ <Borrowed from `transformers`>
        Create a schedule with a learning rate that decreases from the initial lr set in the optimizer to 0,
        after a warmup period during which it increases from 0 to the initial lr set in the optimizer.
        Args:
            optimizer (:class:`~torch.optim.Optimizer`): The optimizer for which to schedule the learning rate.
            num_warmup_steps (:obj:`int`): The number of steps for the warmup phase.
            num_training_steps (:obj:`int`): The total number of training steps.
            last_epoch (:obj:`int`, `optional`, defaults to -1): The index of the last epoch when resuming training.
        Return:
            :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))  # this is correct

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# adapted from
# https://github.com/google-research/fixmatch/blob/d4985a158065947dba803e626ee9a6721709c570/libml/layers.py#L125
class PMovingAverage:
    def __init__(self, nclass, buf_size, device):
        self.ma = torch.ones([buf_size, nclass], device=device, requires_grad=False) / nclass
        #print(self.ma.size(), self.ma)

    def __call__(self):
        v = torch.mean(self.ma, dim=0)
        #print('Printing the updated model probability vector:', v / v.sum())
        return v / v.sum()

    def update(self, entry):
        entry = torch.mean(entry, dim=0, keepdim=True)
        self.ma = torch.cat((self.ma[1:], entry), dim=0)
        #print('Printing the updated model prob matrix:', self.ma, self.ma.size())


# get the number of labels for unequal class distribution
def num_labels_per_class_unequal(whole_train_targets, budget):
    per_class_probs = np.bincount(whole_train_targets) / len(whole_train_targets)
    dists = np.floor(per_class_probs * budget)
    dists[dists == 0] = 1
    left_in_budget = budget - dists.sum()
    if left_in_budget > 0:
        should_have_rounded_up = np.argsort((per_class_probs * budget) - np.floor(per_class_probs * budget))[::-1]
        for i in should_have_rounded_up:
            dists[i] = dists[i] + 1
            left_in_budget -= 1
            if left_in_budget == 0:
                assert dists.sum() == budget
                dists = dists.astype(int)
                break
    return dists


class GenerateResults(object):

    def __init__(self, mod_store):
        # don't bother with results path, just use the model to load results fresh
        # make sure path in standard format

        # either a path or an traintrack object itself will be fed to this class
        # so document accordingly
        if isinstance(mod_store, str):
            if '/models/' not in mod_store:
                mod_store = '/models/' + mod_store
            self.mod_path = mod_store
            self.trainTrackObj = torch.load(self.mod_path)
        else:
            self.mod_path = str(mod_store)
            self.trainTrackObj = mod_store

        self.model_args = self.trainTrackObj.args

        # need to always load model onto this device
        self.homer_device="cuda:0"
        self.model, self.checkpoint = load_model(self.model_args, device=self.homer_device)

        if self.model_args['ema']:
            ema_model = EMA(self.model, 0.999)
            self.model.load_state_dict(self.trainTrackObj.model)
            ema_model.load_state_dict(self.trainTrackObj.ema_model)
            ema_model.apply_shadow()
        else:
            self.model.load_state_dict(self.trainTrackObj.model)

        if self.model_args['dataset'] == 'kenya' or self.model_args['dataset'] == 'serengeti':
            self.names = convert_to_names()
        elif self.model_args['dataset'] == 'cifar':
            self.names = ['plane', 'car', 'bird', 'cat',
                          'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        else:
            self.names = None

        print('Restored model from', mod_store)
        print(f'Trained until {self.trainTrackObj.epoch} with train loss {self.trainTrackObj.train_loss_over_epochs[-1]}\n')

    def get_embeddings(self, test_loader):
        self.embeddingsDict = retrieve_embeddings(model=self.model,
                                         inputs=test_loader, device=self.homer_device)

    def get_test_results(self, test_loader):
        self.testResultsDict = eval_on_test(test=test_loader, model=self.model,
                                       device=self.homer_device, embeddings_only=False)

        self.classesUnique, self.classesCounts = np.unique(self.testResultsDict['targets'], return_counts=True)
        self.classReport = classification_report(self.testResultsDict['targets'],
                                                 self.testResultsDict['preds'], output_dict=True)
        self.test_f1 = f1_score(self.testResultsDict['targets'], self.testResultsDict['preds'], average='macro')
        self.test_accuracy = self.classReport['accuracy']
        print('F1 score:', self.test_f1, '\nAccuracy:', self.test_accuracy)


    def get_loss_curves(self, plot_val=True):
        show_loss_curves(self.trainTrackObj, plot_val=plot_val)

    def accuracy_breakdowns(self, test_loader, produce_plots=True):
        if not hasattr(self, 'testResultsDict'):
            self.get_test_results(test_loader=test_loader)

        # top5 accuracy
        self.test_top5_accuracy = top_k_accuracy_score(self.testResultsDict['targets'],
                                                       self.testResultsDict['outputs'],
                                                       k=5)
        # accuracy by class
        self.cmat = confusion_matrix(self.testResultsDict['targets'], self.testResultsDict['preds'])
        self.testAccuracyByClass = self.cmat.diagonal() / self.classesCounts

        if produce_plots:
            plot_cmat(self.cmat, self.names, len(self.classesCounts))
            plt.barh(self.names, self.testAccuracyByClass)
            plt.title('Accuracy by class')
            plt.tight_layout()
            plt.show()

    def entropy_plot(self, test_loader):
        if not hasattr(self, 'testResultsDict'):
            self.get_test_results(test_loader=test_loader)
        # get top with max uncertainty
        prbslogs = self.testResultsDict['outputs'] * np.log2(self.testResultsDict['outputs'])
        entropy = (0 - np.sum(prbslogs, axis=1)) / np.log2(self.classesUnique)
        self.top_20_highest_entropy_inds = np.argsort(entropy)[-20:]

        # graph which classes have highest average entropy
        # for every entry in the entropy vector there's a corresponding target/prediction
        # use np.where to identify the classes and average the entropy, then plot
        self.testEntropyPerClass = np.zeros(len(self.names))
        for i in range(len(self.names)):
            self.testEntropyPerClass[i] = np.mean(entropy[self.testResultsDict['targets'] == i])
        sortedInds = np.argsort(self.testEntropyPerClass).tolist()
        srtedTestEntropyPerClass = self.testEntropyPerClass[sortedInds]
        sortedNames = [self.names[i] + ', ' + str(i) for i in sortedInds]
        plt.barh(sortedNames, srtedTestEntropyPerClass)
        plt.title('Entropy')
        plt.tight_layout()
        plt.show()

    def accuracy_and_entropy_plots(self, test_loader):
        if not hasattr(self, 'testAccuracyByClass'):
            self.accuracy_breakdowns(self, test_loader=test_loader)
        if not hasattr(self, 'testAccuracyByClass'):
            self.entropy_plot(self, test_loader=test_loader)
        # accuracy and entropy together
        sortedInds = np.argsort(self.testAccuracyByClass).tolist()
        sortedNamesByAccuracy = [self.names[i] + ', ' + str(i) for i in sortedInds]
        y = np.arange(len(sortedNamesByAccuracy))  # the label locations
        width = 0.35  # the width of the bars
        fig, ax = plt.subplots()
        rects1 = ax.barh(y - width / 2, self.testAccuracyByClass[sortedInds], width, label='Accuracy')
        rects2 = ax.barh(y + width / 2, self.testEntropyPerClass[sortedInds], width, label='Entropy')
        ax.set_yticks(y)
        ax.set_yticklabels(sortedNamesByAccuracy)
        ax.legend()
        plt.title('Accuracy and Entropy')
        fig.tight_layout()
        plt.show()

    def visualise_images(self):

        t2he_im_paths = self.testResultsDict['im_paths'][self.top_20_highest_entropy_inds]
        t2he_preds = self.testResultsDict['preds'][self.top_20_highest_entropy_inds].tolist()
        t2he_targets = self.testResultsDict['targets'][self.top_20_highest_entropy_inds].tolist()

        # grid images and report preds/targets; raw images and then as transformed for the model
        if self.model_args['dataset'] == 'kenya':
            data_dir = "/kenya32/"
        elif self.model_args['dataset'] == 'cifar':
            data_dir = "/CIFAR10/"
        else:
            raise Exception('No data target path because seemingly no dataset')
        vis_ims(t2he_im_paths, t2he_targets, t2he_preds, self.model_args,
                data_dir=data_dir)

def convert_to_names(listed=False):
    with open('splits/kenya_animals_dict.json', 'rb') as handle:
        animals_dict = json.load(handle)
    if not listed:
        return animals_dict
    else:
        target = listed.apply(lambda x: animals_dict[x]).to_numpy()
        # target = torch.tensor(target).squeeze()
        return target

def vis_ims(imgs, targets, preds, args, nrow=5, data_dir=None):

    if data_dir is None:
        data_dir=args['data_dir']

    # input is list of image paths, tensor/list of targets, tensor/list of preds
    if torch.is_tensor(targets):
        targets = targets.tolist()
    if torch.is_tensor(preds):
        preds = preds.tolist()

    if type(imgs) is list or type(imgs) is np.ndarray:

        print(f"Showing original images resized to {args['im_res']} by {args['im_res']} without augmentations")
        newlist = []
        for i in range(len(imgs)):
            img = mpimg.imread(data_dir + imgs[i])
            img = transforms.ToTensor()(img)
            img = transforms.Resize((args['im_res'], args['im_res']))(img)
            newlist.append(img)
        plt.imshow(np.transpose(make_grid(newlist, nrow=nrow, padding=5), (1, 2, 0)),
                   interpolation='nearest')
        plt.show()

    elif torch.is_tensor(imgs):

        print("Showing augmented images or images otherwise formatted as a tensor")
        plt.imshow(np.transpose(make_grid(imgs, nrow=nrow, padding=5), (1, 2, 0)),
                   interpolation='nearest')
        plt.show()

    preds = [convert_to_names()[i] for i in preds]
    targets = [convert_to_names()[i] for i in targets]
    [print(f'Image {i} had target: {targets[i]}, prediction: {preds[i]}') for i in range(len(imgs))]


# from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def imshow_mod(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_cmat(cmat, labels_for_plot, nclasses):

    length = len(labels_for_plot)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.tick_params(axis='x', rotation=90)
    ax.set_xticks(range(0, length))
    ax.set_xticklabels(labels_for_plot)
    ax.set_xlabel('Predicted class')
    ax.set_yticks(range(0, length))
    ax.set_yticklabels(labels_for_plot, fontdict={'horizontalalignment': 'right'})
    ax.set_ylabel('True class')
    ax.imshow(cmat, cmap='Blues')

    for i in range(0, nclasses):
        for j in range(0, nclasses):
            #if cmat[i, j] / np.sum(cmat[i, :]) > 0.95:
            if cmat[i, j] > 1000:
                colour = "white"
            else:
                colour = "black"
            ax.text(j, i, cmat[i, j], ha="center", va="center", color=colour)

    # plt.savefig(os.path.join(directory, title))
    fig.tight_layout()
    plt.show()


def show_loss_curves(trainTrackobj, plot_val=True):
    plt.plot(range(trainTrackobj.epoch), trainTrackobj.train_loss_over_epochs, 'bo-', label='Overall train')
    plt.plot(range(trainTrackobj.epoch), trainTrackobj.sup_loss_over_epochs, 'gx-', label='Supervised train')
    plt.plot(range(trainTrackobj.epoch), trainTrackobj.unsup_loss_over_epochs, 'yx-', label='Unsupervised train')
    if plot_val:
        try:
            plt.plot(range(trainTrackobj.epoch), trainTrackobj.val_loss_over_epochs, 'ro-', label='Validation')
        except:
            plt.plot(range(trainTrackobj.epoch - 1), trainTrackobj.val_loss_over_epochs, 'ro-', label='Validation')
        plt.vlines(np.argmin(trainTrackobj.val_loss_over_epochs), colors='k', ymin=0,
                   ymax=max(max(trainTrackobj.train_loss_over_epochs), max(trainTrackobj.val_loss_over_epochs)),
                   linestyles='dashed', label='best model')
    plt.title('Loss curves')
    plt.grid()
    plt.legend()
    plt.show()

    try:
        if torch.is_tensor(trainTrackobj.train_accuracy_over_epochs[0]):
            trainTrackobj.train_accuracy_over_epochs = [i.item() for i in trainTrackobj.train_accuracy_over_epochs]
        plt.plot(range(trainTrackobj.epoch), trainTrackobj.train_accuracy_over_epochs, 'bo-', label='Train')
        try:
            plt.plot(range(trainTrackobj.epoch), trainTrackobj.val_accuracy_over_epochs, 'ro-', label='Validation')
        except:
            plt.plot(range(trainTrackobj.epoch - 1), trainTrackobj.val_accuracy_over_epochs, 'ro-')

        plt.vlines(np.argmax(trainTrackobj.val_accuracy_over_epochs), colors='k',
                   ymin=min(min(trainTrackobj.train_accuracy_over_epochs), min(trainTrackobj.val_accuracy_over_epochs)),
                   ymax=max(max(trainTrackobj.train_accuracy_over_epochs), max(trainTrackobj.val_accuracy_over_epochs)),
                   linestyles='dashed', label='best model')
        plt.title('Accuracy curves')
        plt.grid()
        plt.legend()
        plt.show()
    except:
        pass


def retrieve_fixmatch_outputs(model, dataloader, args, p_data, p_model):

    with torch.no_grad():

        im_ids = []
        pseudo_labels = []
        masks = []
        targets = []
        scores_all = []
        outputs = []

        for i, data in enumerate(dataloader, 0):

            inputs = data['im']
            context_sheet_ids = data['original_id']
            targets.append(data['target'].detach().cpu().numpy())

            inputs = inputs.to(args["device"])
            logits = model(inputs)

            olwp = torch.softmax(logits.detach(), dim=-1)

            # da
            if args['use_da']:
                p_ratio = (1e-6 + p_data) / (1e-6 + p_model())
                outputs_labelled_weak_prob = olwp * p_ratio
                outputs_labelled_weak_prob /= torch.sum(outputs_labelled_weak_prob, dim=1, keepdim=True)
                del p_ratio
            else:
                outputs_labelled_weak_prob = olwp

            scores, p_labels = torch.max(outputs_labelled_weak_prob, dim=-1)
            mask = scores.ge(args['fm_conf_threshold']).float()

            im_ids.append(context_sheet_ids)
            masks.append(mask.detach().cpu().numpy())
            pseudo_labels.append(p_labels.detach().cpu().numpy())
            scores_all.append(scores.detach().cpu().numpy())
            outputs.append(outputs_labelled_weak_prob.detach().cpu().numpy())

        ids = np.concatenate(im_ids, axis=None)
        masks = np.concatenate(masks, axis=0)
        pseudo_labels = np.concatenate(pseudo_labels, axis=0)
        targets = np.concatenate(targets, axis=None)
        scores_all = np.concatenate(scores_all, axis=None)
        outputs = np.concatenate(outputs, axis=0)

    return ids, pseudo_labels, targets, masks, scores_all, outputs


class CombineALResults(object):
    # basically this class is a way to standardise the results outputs
    # so that they can be compared on the test set

    def __init__(self, rootpath, test_loader):

        if '/models/' not in rootpath:
            rootpath = '/models/' + rootpath

        self.rootpath = rootpath
        self.mod_paths = []
        self.test_acc = []
        self.test_top5_acc = []
        self.test_f1 = []
        self.test_precision = []
        self.test_recall = []
        self.test_acc_by_class = []
        self.test_f1_prec_recall_by_class = []
        self.cmat = []
        self.newQueries_at_iter = []
        self.p_data_at_iter = []
        self.p_model_at_iter = []

        if 'ActiveLearningTrackObject' in rootpath:
            self.obj_type = 'ALTrackObject'
            if '.pt' not in rootpath:
                rootpath = rootpath + '.pt'
            ALobject = torch.load(rootpath)
            if hasattr(ALobject, 'nCycledThrough'):
                self.nCycledThrough = []
                for cycledList in ALobject.nCycledThrough:
                    self.nCycledThrough.append(len(cycledList))
            if len(ALobject.trainTracksOverAL) == 20:
                for trackobject in ALobject.trainTracksOverAL:
                    item = GenerateResults(trackobject)
                    self.append_things(item, test_loader)
            else:
                assert len(ALobject.trainTracksOverAL) == 21
                for trackobject in ALobject.trainTracksOverAL[1:]:
                    item = GenerateResults(trackobject)
                    self.append_things(item, test_loader)
            self.mod_paths = [i for i in range(20)]
            self.newQueries_at_iter = ALobject.newQueries
            self.args = ALobject.argsForThisRun
            if 'proportional_per_class2' in rootpath:
                self.split = 2
            elif 'proportional_per_class3' in rootpath:
                self.split = 3
            else:
                self.split = 1

        else:
            self.obj_type = 'CPs'
            self.split = 1
            for i in np.arange(132, 513, 20):
                path = rootpath + str(i) + '.pt'
                if i == 132:
                    try:
                        item = GenerateResults(path)
                    except:
                        if 'true_uniform_random_' in rootpath:
                            print('Using PT model based on the TRUE RAND INIT random basic run as 132 run')
                            item = GenerateResults(pathforALres1)
                        else:
                            print('Using PT model based on the random basic run as 132 run: this should be split1 mods only')
                            item = GenerateResults(pathforALres2)
                    self.args = item.trainTrackObj.args
                else:
                    item = GenerateResults(path)
                self.mod_paths.append(path)
                self.append_things(item, test_loader)

    def append_things(self, item, test_loader):
        item.accuracy_breakdowns(test_loader, produce_plots=False)
        self.test_acc.append(item.test_accuracy)
        self.test_top5_acc.append(item.test_top5_accuracy)
        self.test_f1.append(item.test_f1)
        self.test_precision.append(item.classReport['macro avg']['precision'])
        self.test_recall.append(item.classReport['macro avg']['recall'])
        self.test_acc_by_class.append(item.testAccuracyByClass)
        self.cmat.append(item.cmat)
        self.test_f1_prec_recall_by_class.append(item.classReport)
        self.p_data_at_iter.append(item.trainTrackObj.p_data.cpu().numpy())
        self.p_model_at_iter.append(np.mean(item.trainTrackObj.p_model.ma.cpu().numpy(), axis=0))