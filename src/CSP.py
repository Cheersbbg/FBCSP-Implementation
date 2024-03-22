from src import preprocessing
import scipy
import numpy as np

def get_class_covariance_matrix(trials):
    """Gets the average covariance matrix of a set of motor imagery trials for one class of data

    Parameters:
    trials: A Numpy Array of shape (# of Trials) x (# Of Channels) # (Length Per Trial)

    Returns:
    A CxC covariance matrix, where C is the number of channels
    """

    #Allocating space for covariance matrix
    n_trials,n_channels,trial_length = trials.shape
    sum_of_covariance_matrices = np.zeros((n_channels,n_channels)) 

    for X in trials:

        #Covariance matrix of current trial normalized by its trace
        trial_covariance_matrix = X @ X.T
        sum_of_covariance_matrices += trial_covariance_matrix / np.trace(trial_covariance_matrix)

    average_covariance_matrix = sum_of_covariance_matrices / n_trials

    return average_covariance_matrix


def get_spatial_filters(trials,labels,
                        n_csp = 1,return_eigvals = False):
    
    """Returns the pair of spatial filters associated with the GEVD problem in CSP

    Parameters:
    trials: A Numpy Array of shape (# of Trials) x (# Of Channels) # (Length Per Trial)
    
    labels: The label of each trial in trials

    n_csp: The number of pairs of spatial filters to use

    return_eigvals: Whether or not to return the eigenvalues associated with the GEVD Problem
    
    Returns:
    A matrix of Shape 2*n_csp*C, where C is the number of channels in trials

    """
    negative_trials,positive_trials = preprocessing.get_pos_and_neg_class_trials(trials,labels)


    S_pos = get_class_covariance_matrix(positive_trials)
    S_neg = get_class_covariance_matrix(negative_trials)

    
    eigenvalues,eigenvectors = scipy.linalg.eigh(S_pos,S_pos + S_neg)

    #Sorts eigenvectors from GEVD based on eigenvalues in descending order
    sorted_idxs = np.argsort(eigenvalues)[::-1]

    eigenvectors = eigenvectors[:,sorted_idxs]

    #Getting spatial filters from the first n_csp columns and last n_csp columns of the eigenvectors
    spatial_filters = np.hstack((eigenvectors[:,:n_csp],eigenvectors[:,-n_csp:]))

    if return_eigvals:
        return spatial_filters,eigenvalues[sorted_idxs]

    return spatial_filters

def map_trials_to_csp_space(trials,spatial_filters):

    """Maps set of trials in the original data space to the CSP Space

    Parameters:
    trials: A Numpy Array of shape (# of Trials) x (# Of Channels) # (Length Per Trial)

    spatial_filters: The spatial filters associated with CSP. Obtained from running get_spatial_filters

    Returns: A numpy array of shape (# of Trials x #CSP Channels x # Length Per Trial)
    """

    csp_trials = spatial_filters.T @ trials

    return csp_trials


def get_filter_bank_spatial_filters(filter_bank_trials,
                                        labels, n_csp = 2):

    """Gets the spatial filters associated with data decomposed into multiple different frequency ranges

    Parameters:

    filter_bank_trials: A numpy array of shape (# of frequency bands x # Of Trials x # Channels x # Length per trial)

    labels: Labels associated with the filter bank trials

    n_csp: The number of csp pers for each spatial filter on each frequency band of the filter bank


    Returns:
    The spatial filters for the data across each frequency band in the filter bank
    """

    
    get_band_spatial_filters = lambda f_band_idx:get_spatial_filters(filter_bank_trials[f_band_idx],labels,n_csp = n_csp)
    
    spatial_filters_across_bands = np.array(list(map(get_band_spatial_filters,np.arange(len(filter_bank_trials)))))

    return spatial_filters_across_bands


def get_log_var_csp_features(trials,W):
    """Obtains log variance based features for CSP Filtered Data

    Parameters:
    trials: A Numpy Array of shape (# of Trials) x (# Of Channels) # (Length Per Trial)

    W:Learned Spatial spatial filters from get_spatial_filters function

    Returns: The log variance based features for a set of motor imagery trials
    
    """

    n_trials = len(trials)
    n_chans_csp = W.shape[-1]

    log_var_features = np.ones((n_trials,n_chans_csp))

    for i,X in enumerate(trials):

        #Covariance matrix for the current trial in CSP Space
        csp_covariance_matrix = W.T @ X @ X.T @ W

        #Already trace normalized, so not doing it again, just taking log of diagonal elements
        log_csp_cov = np.log(np.diag(csp_covariance_matrix))

        log_var_features[i] = log_csp_cov

    return log_var_features



def get_log_var_fbank_csp_features(trials,band_spatial_filters):

    """Gets the log variance based features for trials decomposed into multiple different frequency bands

    Parameters:
    trials: A Numpy Array of shape (# of Trials) x (# Of Channels) # (Length Per Trial)

    band_spatial_filters: The spatial filters associated with each frequency band in the filter bank
    obtained with get_filter_bank_spatial_filters function
    
    """

    n_freq_bands,n_trials = trials.shape[:2]
    n_chans_csp = band_spatial_filters.shape[-1]

    #Allocating space for the log variance features
    log_var_features = np.ones((n_trials,n_chans_csp * n_freq_bands))

    #Obtaining the log variance features across each frequency band and then updating
    #The corresponding section of the feature vectors for that particular frequency band
    for i in np.arange(n_freq_bands):

        cur_band_trials = trials[i]
        cur_band_spatial_filter = band_spatial_filters[i]

        cur_log_var_features = get_log_var_csp_features(cur_band_trials,cur_band_spatial_filter)
        
        log_var_features[:,i*n_chans_csp:(i+1)*n_chans_csp] = cur_log_var_features

    return log_var_features


    

    
    
    
        






