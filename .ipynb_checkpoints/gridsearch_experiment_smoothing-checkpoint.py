from src import augmentation,cleaning,loaders,classifier,preprocessing

from sklearn.model_selection import KFold
from sklearn.svm import SVC

import numpy as np

from tqdm import tqdm

from argparse import ArgumentParser
import sys

import json
import os


parser = ArgumentParser()

parser.add_argument("--subject",help="subject to run gridsearch experiment on.")
parser.add_argument("--parameter_grid",help = "JSON File storing the parameter grid to search on during experiments.")
parser.add_argument("--save_every",help = "How often the grid search results should be saved")
parser.add_argument("--save_checkpoints_to",help = "The directory that checkpoints are saved to.")
parser.add_argument("--resume_from",help = "Iteration of the gridsearch to resume code from if program crashed or stalled")


args = parser.parse_args().__dict__

source_reject_per_subject = [[0,1,3]]

if not args['resume_from']:
    args['resume_from'] = None

if not args['subject']:
    print('Subject number was not provided... exiting program')
    sys.exit(1)

if 'pg' not in args:
    args.update({'pg':None})

if 'save_every' not in args:
    args.update({'save_every':None})

if 'save_checkpoints_to' not in args:
    args.update({'save_checkpoints_to':None})

if args['save_every'] and not args['save_checkpoints_to']:
    print('Provided a regular interval to save gridsearch results as checkpoints, but not a directory to save them to. Exiting program.')

    sys.exit(1)

save_every = int(args['save_every'])
checkpoint_dir = args['save_checkpoints_to']

if not os.path.isdir(checkpoint_dir):
    parent_dir = os.getcwd()
    os.makedirs(os.path.join(parent_dir,checkpoint_dir))

subject_no = int(args['subject'])
print(f'Running experiments for subject {subject_no}')


n_ica_components = 10
chs_to_exclude = source_reject_per_subject[subject_no - 1]
print(f'Running ICA Decomposition to clean data with {n_ica_components} components. Rejecting ICs {chs_to_exclude}. Please be patient this may take a little bit. \n')

#By default does this for the training data in fit ica since we don't touch the evaluation data at all
ica_object,subject_recording = cleaning.fit_ica(subject_no,n_ica_components)

ica_object.exclude = chs_to_exclude
ica_object.apply(subject_recording);


#When getting test fold accuracy we only want to evaluate on the noisy, uncleaned data, because
#this is the same state that the evaluation data will be in
cleaned_trials,cleaned_labels = loaders.extract_motor_imagery_trials(recording = subject_recording)
original_trials,original_labels = loaders.extract_motor_imagery_trials(subject_no = subject_no)

n_trials = len(cleaned_trials)

print(f'Extracted cleaned trials. Total of {n_trials} original motor imagery trials')

if args['pg']:
    parameter_grid_json = args['pg']

if not args['pg']:
    parameter_grid_json = 'grid search results/parameter_grid_smoothing.json'
    print(f'Path to parameter combination was not provided, using default parameter grid for smoothing experiments at {parameter_grid_json}')

    if not os.path.isfile(parameter_grid_json):
        print(f'Path to parameter grid json was not provided, and default path not on directory. Exiting program')
        sys.exit(1)

parameter_combinations = preprocessing.generate_parameter_combinations_from_grid(parameter_grid_json)

n_combinations = len(parameter_combinations)
print(f'Getting parameter combinations from provided grid... {n_combinations} total combinations\n')


f_bands = [[f,f+4] for f in range(4,40,4)]
fs = 250

filter_bank_coefficients = preprocessing.compute_filter_bank_coefficients(fs,f_bands)

test_accuracies_dict = {}

best_accuracy = -1
optimal_parameter_combination = None

if args['resume_from']:
    parent_dir = os.getcwd()
    checkpoint_dir = checkpoint_dir

    resume_from = args['resume_from']
    
    checkpoint_json = f'checkpoint_subject1_gridsearch_iter_{resume_from}.json'

    checkpoint_path = os.path.join(parent_dir,checkpoint_dir,checkpoint_json)
    
    if not os.path.isfile(checkpoint_path):
        print(f'This checkpoint does not exist. Exiting program.')
        sys.exit(1)

    with open(checkpoint_path) as json_file:
        resume_from_accuracy_dict = json.load(json_file)
        test_accuracies_dict = resume_from_accuracy_dict

if args['resume_from']:
	print(f'Skipping to iteration {resume_from}. Will not be reflected on the progress bar indicator until the next checkpoint.')
for i,parameter_combination in enumerate(tqdm(parameter_combinations)):
    if args['resume_from']:
    	if i < int(resume_from):
        	continue

    #First entry in the pair for sigma for A1 smoothing, second one for D1 smoothing
    C,percent_increase_dataset,sigma_pair,smooth_on = parameter_combination

    if smooth_on == 'A1':
        sigma = sigma_pair[1]
    else:
        sigma = sigma_pair[0]

    #10 Fold Cross CV
    kf = KFold(n_splits = 10)
    test_accs = []
    train_accs = []

    train_test_indices = kf.split(np.arange(len(cleaned_labels)),cleaned_labels)

    for train_indices,test_indices in train_test_indices:

        #Again, train on clean data, test on the natural noisy data to see if our augmentation 
        #can actually account for this noise
        train_trials,test_trials = cleaned_trials[train_indices],original_trials[test_indices]
        train_labels,test_labels = cleaned_labels[train_indices],original_labels[test_indices]

        n_trials_to_augment = int(len(train_trials) * percent_increase_dataset)
        
        #Includes the original trials and labels as well - augment ONLY after we split the train and test
        #data in each fold, to make sure that augmented versions of test data does not leak into the
        #training set and inflate mean test fold accuracy
        augmented_trials,augmented_labels = augmentation.wavelet_smoothing(train_trials,
                                                          train_labels,
                                                          sigma,
                                                          smooth_on,
                                                          n_trials_to_augment)

        augmented_trials_fb = preprocessing.filter_bank_trials(augmented_trials,fs,filter_bank_coefficients)
        test_trials_fb = preprocessing.filter_bank_trials(test_trials,fs,filter_bank_coefficients)

        base_classifier = SVC(kernel = 'linear',C = C,probability = True)

        classifiers,spatial_filters,mibif_indices = classifier.train_classifier(augmented_trials_fb,
                                                                                augmented_labels,
                                                                                base_classifier = base_classifier,
                                                                                mode = 'MIBIF')

        
        test_predictions = classifier.make_predictions(test_trials_fb,
                                                       classifiers,
                                                       spatial_filters,
                                                       mode = 'MIBIF',
                                                       mibif_indices = mibif_indices)

        test_acc = np.sum(test_predictions == test_labels) / len(test_labels)
        test_accs.append(test_acc)

    mean_test_acc = np.mean(test_accs)
    
    if mean_test_acc > best_accuracy:
        best_accuracy = mean_test_acc
        optimal_parameter_combination = parameter_combination

    keys_as_string = ' '.join([str(parameter) for parameter in parameter_combination])
    test_accuracies_dict.update({(keys_as_string):mean_test_acc})

    if i % save_every == 0:

        parent_dir = os.getcwd()
        save_dir = os.path.join(parent_dir,checkpoint_dir)
        
        checkpoint_name = f'checkpoint_subject{subject_no}_gridsearch_iter_{i}.json'
        save_to = os.path.join(save_dir,checkpoint_name)

        print(f'\n Saving checkpoint to {save_to} at iteration {i}')

        with open(save_to,'w') as checkpoint_file: 
            json.dump(test_accuracies_dict, checkpoint_file)

    tqdm.write(f'{i}/{len(parameter_combinations)}', end='\r')

