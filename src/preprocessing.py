import scipy
import numpy as np
import concurrent.futures

import os
import json

import itertools

# When filtering returning the trials from 0.5s to 2.5s (relative to the onset of the cue)
# Using chebyshev type II Zero Phase bandpass filter
def bandpass_filter_trials(trials,fs,l_freq,h_freq,rp = 30,order = 3,t_start = 0.5,t_end = 2.5):

    """Bandpass filters the trials
    Parameters:
    trials: A Numpy Array of shape (# of Trials) x (# Of Channels) # (Length Per Trial)

    fs: Sampling frequency of trials

    l_freq: low cut-off frequency of filter

    h_freq: High cut-off frequency of filter

    rp: Filter passband ripple

    order: order of filter

    t_start: Start of motor imagery relative to cue onset in recording

    t_end: Start of motor imagery relative to cue onset in recording 

    Returns:Bandpass filtered trials and extracts most relevant section of motor imagery trial
    to use in classification
    """

    nyquist = fs/2
    
    normalized_l_freq = l_freq / nyquist
    normalized_h_freq = h_freq / nyquist

    b, a = scipy.signal.cheby2(order, rp,[normalized_l_freq, normalized_h_freq], 'bandpass')

    filtered_trials = scipy.signal.filtfilt(b,a,trials)

    #Since originally we starting extracting trials 4.5 seconds before the onset of a cue
    trial_start = int(fs * (4.5 + t_start))
    trial_end = int(fs*(4.5 + t_end))
    
    return filtered_trials[:,:,trial_start:trial_end]


#Pre-computing filter coefficients so that this does not need to be repeated, especially important when
#doing KFold CV and gridsearch to save a lot of time
def compute_filter_bank_coefficients(fs,freq_bands,rp = 30,order = 3):
    """Computes filter coefficients for each filter in a filter bank. 

    Parameters:
    fs: Sampling frequency of recorded trials

    freq_bands: Frequency bands to use in the filter bank

    rp: Filter passband ripple
    
    order: Order of filter    
    """
    
    filter_coefficients = []
    nyquist = fs/2
    
    for f_band in freq_bands:
        normalized_l_freq = f_band[0] / nyquist
        normalized_h_freq = f_band[1] / nyquist

        b, a = scipy.signal.cheby2(order, rp,[normalized_l_freq, normalized_h_freq], 'bandpass')

        filter_coefficients.append((b,a))

    return filter_coefficients


def filter_bank_trials(trials, fs, filter_coefficients, t_start=0.5, t_end=2.5):

    """Decomposes filters into multiple frequecny bands based on a filter bank

    Parameters
    trials : A Numpy Array of shape (# of Trials) x (# Of Channels) # (Length Per Trial)

    fs: Sampling frequency of data during recordig

    filter_coefficients: Filter coefficients for each filter in the filter bank

    t_start: Start of motor imagery relative to cue onset in recording
    
    t_end: Start of motor imagery relative to cue onset in recording 

    Returns:
    A numpy array of shape (# of frequency bands x # Of Trials x # Channels x # Length per trial)
    """
    
    n_trials, n_chans = trials.shape[:2]
    n_freq_bands = len(filter_coefficients)
    new_trial_length = int(fs * (t_end - t_start))
    filter_bank_trials = np.empty((n_freq_bands, n_trials, n_chans, new_trial_length))

    trial_start = int(fs * (4.5 + t_start))
    trial_end = int(fs * (4.5 + t_end))

    def filter_band_trial(args):
        b, a = args
        filtered_trials = scipy.signal.filtfilt(b, a, trials)[:, :, trial_start:trial_end]
        return filtered_trials

    # Prepare arguments for concurrent processing
    args_list = filter_coefficients

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(filter_band_trial, args_list))

    for i, result in enumerate(results):
        filter_bank_trials[i] = result

    return filter_bank_trials


#Shuffling of train trials done during training to alleviate issue of classifier just learning
#temporal dependencies of training data and then giving misleading accuracy scores on unseen evaluation sets
def shuffle_trials_and_labels(trials,labels):

    """Shuffles trials and labels together

    Parameters
    trials : A Numpy Array of shape (# of Trials) x (# Of Channels) # (Length Per Trial) 

    labels: label of each trial, i.e which motor imagery was executed
    
    """

    label_indices = np.arange(len(labels))
    np.random.shuffle(label_indices)

    labels = labels[label_indices]

    #for filter bank trials
    if trials.ndim == 4:
        trials = trials[:,label_indices]
    else:
        trials = trials[label_indices]


def create_ovr_dataset(labels):

    """Creates OneVersusRest dataset given a set of labels

    Parameters:
    labels: The original multi-class labels of each of the trials

    Returns:
    A set of labels associated with each sub binary classification task in a OneVersusRest classifier.
    Will be a numpy array of shape (# Classes, length of labels)

    """


    #Each entry will be the labels associated with each dataset in an OvR dataset. The trials themselves do not change, hence,
    #to be memory efficient we will not create four copies of the dataset.
    ovr_labels = []

    n_classes = len(set(labels))

    for class_idx in range(n_classes):
            
        #The labels that are not "class_idx"
        cur_ovr_labels = np.array([0 if label == class_idx else 1 for label in labels])
        ovr_labels.append(cur_ovr_labels)

    return ovr_labels

#Useful in the CSP.py code
def get_pos_and_neg_class_trials(trials,labels):

    """Splits the trials according to their label, assuming a binary label

    Parameters
    trials : A Numpy Array of shape (# of Trials) x (# Of Channels) # (Length Per Trial) 

    labels: label of each trial, i.e which motor imagery was executed

    Returns:
    A tuple where the first element represent the positive trials (class 0),
    and the second element represents the negative trials (class 1)
    """

    positive_trials = trials[np.where(labels == 0)]
    negative_trials = trials[np.where(labels == 1)]

    return positive_trials,negative_trials

