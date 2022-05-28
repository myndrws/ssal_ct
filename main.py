##################################################
# Purpose: Script to set and run other scripts   #
# Author: Amy Andrews                            #
# Resources used:
# Pytorch documentation https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# Pytorch documentation https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
##################################################
import torch.cuda
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.backends import cudnn
from torch.utils.data import DataLoader, Subset
import random
import pprint

from load_data import load_data
from set_args import get_args
from utils import *
from train_loop import train_model


def main(args):

    # seed setting for repro
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False

    device = torch.device(args['device'])

    # model_bt for model before training, just 'model' after
    if 'pretrained_model' not in args or args['pretrained_model'] == '':
        model_bt, checkpoint = load_model(args)
    elif args['pretrained_model'] != '':
        path = args['mod_dir'] + args['pretrained_model']
        model_bt, checkpoint = load_model(args, load_from_path=path)

    # training loop for different scenarios
    if args['train_loss'] == 'supervised':
        train_set, val_set, test_set = load_data(args)
        train = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'], pin_memory=True)
        val = DataLoader(val_set, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'], pin_memory=True)
        print('Train set length:', len(train_set))
    elif args['train_loss'] == 'fixmatch':
        supervised_set, unsupervised_set, val_set, test_set = load_data(args)
        # set batch size for unlabelled data using fixmatch arguments
        unsupervised_batch_size = args['batch_size'] * args['fm_ratio_mu']
        supervised = DataLoader(supervised_set, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'],
        drop_last=True, pin_memory=True)
        unsupervised = DataLoader(unsupervised_set, batch_size=unsupervised_batch_size, shuffle=True, num_workers=args['num_workers'],
        drop_last=True, pin_memory=True)
        val = DataLoader(val_set, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'], pin_memory=True)
        train = [supervised, unsupervised]

    opt = optim.SGD(model_bt.parameters(), lr=args['learning_rate'],
                    weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)
    if args['scheduler'] == 'cosine':
        # from https://github.com/Celiali/FixMatch/blob/main/experiments/experiment.py
        scheduler = get_cosine_schedule_with_warmup(optimizer=opt, num_warmup_steps=0,
                                                    num_training_steps=2**20)  # hard coded rather than num iters for replication
    elif args['scheduler'] == 'plateau' or args['scheduler'] == 'plateau_train':
        scheduler = lr_scheduler.ReduceLROnPlateau(opt, 'min', verbose=True)

    # set paths and names
    args = assign_paths(args)

    # run
    model, ema_model, res = train_model(model_bt, opt, scheduler, train, val, args, checkpoint=checkpoint)

    # report where model and results are stored
    print(f'Model stored at {args["mod_path"]} after {res["total_epochs_trained"]} epochs')


# run if running this file
if __name__ == '__main__':

    args = get_args(bash_parser=True)

    print('\n**********************************')
    print('Experiment :', args['train_loss'] + '_' + args['mod_type'] + '_' + args['scheduler'])
    print('Dataset    :', args['dataset'])
    pprint.pprint(args)
    print('************************************')

    main(args)