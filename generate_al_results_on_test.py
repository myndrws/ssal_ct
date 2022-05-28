######################################################
# Purpose: Generate fixmatch active learning results #
# Author: Amy Andrews                                #
######################################################

import os
import argparse
from utils import CombineALResults
from load_data import load_data, get_serengeti_test
from set_args import get_args
import pickle
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Active Learning with SSL')
parser.add_argument('--dataset', default='kenya', choices=['kenya', 'serengeti'], type=str)
parser.add_argument('--list_of_dirs', default=['al_random_basic', 'al_random_rejection', 'al_kcenter_basic',
                                               'al_kcenter_rejection', 'al_entropy', 'al_entropy_rejection',
                                               'al_custom_basic', 'al_custom_rejection', 'al_margin_basic',
                                               'al_margin_rejection', 'al_kcenter_hybrid_basic',
                                               'al_kcenter_hybrid_rejection'])
results_gen_args = vars(parser.parse_args())

args = get_args()  # args don't really matter for test set
if results_gen_args['dataset'] == 'kenya':
    supervised_set, unsupervised_set, val_set, test_set = load_data(args)
    test = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)
else:
    assert results_gen_args['dataset'] == 'serengeti', 'You must choose between kenya and serengeti'
    # if you want to use this further, you have to get a dictionary of the animals and their corresponding labels
    args['dataset'] = 'serengeti'
    args['data_dir'] = ""
    args['metadata'] = args['data_dir'] + args['dataset'] + '_context_final.csv'
    test_set = get_serengeti_test(args)
    test = DataLoader(test_set, batch_size=200, shuffle=False, num_workers=6)

list_of_dirs = results_gen_args['list_of_dirs']
dataset = results_gen_args['dataset']


# list_of_dirs = ['al_entropy_rejection']
# dataset = 'kenya'

# in the da just need to select species in [list of species interested in]
# then need to map these species to the 'convert to names' dict indices

############################################################
# read in AL objects, extract kenya info, pickle in results
############################################################

class saveALResultsOnly():
    def __init__(self, ALresults, name):
        self.name = name
        self.rootpath = ALresults.rootpath
        self.split = ALresults.split
        self.args = ALresults.args
        self.mod_paths = ALresults.mod_paths
        self.newQueries_at_iter = ALresults.newQueries_at_iter
        self.p_data_at_iter = ALresults.p_data_at_iter
        self.p_model_at_iter = ALresults.p_model_at_iter
        self.test_acc = ALresults.test_acc
        self.test_top5_acc = ALresults.test_top5_acc
        self.test_f1 = ALresults.test_f1
        self.test_precision = ALresults.test_precision
        self.test_recall = ALresults.test_recall
        self.test_acc_by_class = ALresults.test_acc_by_class
        self.cmats = ALresults.cmat
        self.test_f1_prec_recall_by_class = ALresults.test_f1_prec_recall_by_class
        if hasattr(ALresults, 'nCycledThrough'):
            self.n_cycled_through = ALresults.nCycledThrough


# loop through all the dirs
# if the files don't have the words 'ActiveLearningTrackObject' in them but do have 'CPforALatEpoch' in them
# select one of these and get rid of everything after CPforALatEpoch
# run the CombineALResults for this root path and check the length of it
# then in the same dir check for the ActiveLearningTrackObjects, there should be two
# then run these as well
# then based on the name of the directory and the name of the split and 'kenya_test'
# pickle the objects in the results/al_loops dir
for al_dir in list_of_dirs:
    print('Looking in', al_dir, '...')
    dir_path = al_dir
    name_root = al_dir + '_' + dataset + '_test_results.pkl'
    all_files = os.listdir(dir_path)
    al_track_objects = [file for file in all_files if 'ActiveLearningTrackObject' in file]
    none_al_track_objects = [file for file in all_files if 'ActiveLearningTrackObject' not in file]
    assert len(all_files) == len(al_track_objects) + len(none_al_track_objects)
    # choosing files without 'proportional_per_class_' deselects files if not 'proportional_per_class[split_num]'
    al_track_objects = [file for file in al_track_objects if 'proportional_per_class_' not in file]

    if len(al_track_objects) == 3:
        split1_root_path = [file for file in al_track_objects if 'proportional_per_class1_' in file]
        assert len(split1_root_path) == 1
        split1_root_path = split1_root_path[0]
        all_relevant_paths = al_track_objects
        print('All the relevant paths here are track objects')
    else:
        assert len(al_track_objects) == 2, 'Check the AL track objects in this dir'
        split1_root_path = none_al_track_objects[0].split('CPforALatEpoch', 1)[0] + 'CPforALatEpoch'
        all_relevant_paths = al_track_objects
        all_relevant_paths.append(split1_root_path)

    print('Paths are', all_relevant_paths, '\nGenerating results...')
    for al_models in all_relevant_paths:
        results = CombineALResults(al_dir + '/' + al_models, test)
        resultsToSave = saveALResultsOnly(results, name_root)
        if al_models == split1_root_path:
            pkl_name = 'results/al_loops/split1_' + name_root
        elif 'proportional_per_class2' in al_models:
            pkl_name = 'results/al_loops/split2_' + name_root
        elif 'proportional_per_class3' in al_models:
            pkl_name = 'results/al_loops/split3_' + name_root
        else:
            raise Exception('unknown split')
        with open(pkl_name, 'wb') as handle:
            pickle.dump(resultsToSave, handle, pickle.HIGHEST_PROTOCOL)
        handle.close()
        print('Results generated for', al_models, '\nand saved at', pkl_name)

print('All finished!')
