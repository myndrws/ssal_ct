##################################################
# Purpose: Set args for use in scripts           #
# Author: Amy Andrews                            #
##################################################

import argparse

def get_args(bash_parser=False):

    if bash_parser:
        parser = argparse.ArgumentParser(description='Active Learning with SSL')

        # dataset
        parser.add_argument('--dataset', default='cifar', choices=['cct20', 'kenya', 'cifar'], type=str)
        parser.add_argument('--ten_perc_split', default=False, type=bool)
        parser.add_argument('--10_perc_subset_id', choices=range(9), type=int)
        parser.add_argument('--labelling_split_type', default='equal_per_class',
                            choices=['equal_per_class', 'proportional_per_class', 'true_uniform_random'])
        parser.add_argument('--im_res', default=32, choices=[32, 112], type=int)
        parser.add_argument('--n_available_labels', default=4000, type=int)
        parser.add_argument('--strong_augment', default='RandAugment', type=str)
        parser.add_argument('--1_perc_subset_id', type=int, choices=[1,2,3,4,5,6,7])
        parser.add_argument('--split_id', type=int, choices=[1,2,3,4,5])

        # model
        parser.add_argument('--pretrained_model', default='', type=str)
        parser.add_argument('--imnet_pretrained', default=False, type=bool)
        parser.add_argument('--projection_dims', type=int)
        parser.add_argument('--train_loss', default='fixmatch', choices=['supervised', 'fixmatch'], type=str)
        parser.add_argument('--mod_type', default='wrn28', choices=['wrn28', 'rn18'], type=str)
        parser.add_argument('--ema', default=True, type=bool)

        # optimisation
        parser.add_argument('--learning_rate', default=0.03, type=float)
        parser.add_argument('--weight_decay', default=0.0005, type=float)
        parser.add_argument('--momentum', default=0.9, type=float)
        parser.add_argument('--scheduler', default='cosine', choices=['plateau', 'plateau_train', 'cosine'])
        parser.add_argument('--patience', default=2**20, type=int)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--n_total_iters', default=2**20, type=int)
        parser.add_argument('--n_iters_per_epoch', default=1024, type=int)

        # fixmatch
        parser.add_argument('--fm_conf_threshold', default=0.95, type=float, help='threshold for pseudo-labelling, aka tau')
        parser.add_argument('--fm_ratio_mu', default=7, type=int, help='ratio of unlabelled data to labelled data per iteration')
        parser.add_argument('--fm_loss_weight_lmda', default=1, type=int, help='weight for unsupervised loss')
        parser.add_argument('--use_da', default=False, action='store_true', help='use distribution alignment as part of training')

        # active learning
        parser.add_argument('--query_strat', default='random', choices=['random', 'kcenter', 'custom',
                                                                        'least_certain', 'margin', 'kcenter_hybrid',
                                                                        'predetermined_1600_proportional',
                                                                        'predetermined_1600_true_rand'])
        parser.add_argument('--n_queries_per_al_iteration', default=80, type=int)
        parser.add_argument('--al_iteration_every_n_epochs', default=20, type=int)
        parser.add_argument('--using_al', default=False, action='store_true')
        parser.add_argument('--use_rejection_sampling', default=False, action='store_true')

        # directories and paths
        parser.add_argument('--base_dir', default='', type=str)
        parser.add_argument('--mod_dir', default='', type=str)
        parser.add_argument('--results_dir', default='', type=str)
        parser.add_argument('--data_dir', default='', type=str)
        parser.add_argument('--metadata', default='', type=str)

        # misc
        parser.add_argument('--seed', default=42, type=int)
        parser.add_argument('--num_workers', default=6, type=int)
        parser.add_argument('--device', default="cuda", choices=["cuda:0", "cuda:2", "cuda"], type=str)
        parser.add_argument('--exp_notes', default='', type=str)
        parser.add_argument('--using_val_to_train', default=False, type=bool)

        # turn into a dictionary
        args = vars(parser.parse_args())

    else:
        args = {"dataset": 'kenya'}
        args['patience'] = 2**20
        args['batch_size'] = 64
        args['train_loss'] = 'fixmatch'
        args['mod_type'] = 'wrn28'
        args['seed'] = 42
        args['imnet_pretrained'] = False  # false for CIFAR test
        args['pretrained_model'] = ''  # set this to the location of the model to continue training from if required
        args['scheduler'] = 'cosine'
        args['n_available_labels'] = 320
        args['1_perc_subset_id'] = 7
        args['split_id'] = 0
        args['labelling_split_type'] = 'proportional_per_class'   # or proportional_per_class or true_uniform_random
        args['projection_dims'] = 10 if args['dataset'] == 'cifar' else 20
        args['query_strat'] = 'kcenter_hybrid'
        args['strong_augment'] = 'RandAugment'  # if this is set at all this is what will be using in training
        args['fm_conf_threshold'] = 0.95
        args['fm_ratio_mu'] = 7 # so total batch size would be 64 labelled + 448 strong unlabelled + 448 weak unlabelled
        args['fm_loss_weight_lmda'] = 1
        args['using_val_to_train'] = False
        args['ema'] = True
        args['n_total_iters'] = 524288 #2**20  #524288 is half original
        args['n_iters_per_epoch'] = 1024
        args['use_da'] = True
        args['exp_notes'] = ''

        args['learning_rate'] = 0.03
        args['weight_decay'] = 0.0005
        args['momentum'] = 0.9
        args['num_workers'] = 6
        args['device'] = "cuda:0"
        args['im_res'] = 32 #if args['dataset'] == 'cifar' else 112

        args['n_queries_per_al_iteration'] = 80
        args['al_iteration_every_n_epochs'] = 20
        args['using_al'] = False
        args['use_rejection_sampling'] = False

        # directories and paths
        args['base_dir'] = ''
        args['mod_dir'] = ''
        args['results_dir'] = args['base_dir'] + "results/"
        args['data_dir'] = '' if args['dataset'] == 'cifar' else ""
        args['metadata'] = args['data_dir'] + args['dataset'] + '_context_final.csv'

    # set through iters and iters per epoch
    args['epochs'] = args['n_total_iters'] // args['n_iters_per_epoch']

    return args


