from src import CSP,preprocessing,MIBIF,feature_extraction
import numpy as np

import copy

def train_classifier(trials,labels,base_classifier,mode = 'CSP', n_csp = 2,n_mibif_pairs = 4):

    fs = 250
    l_freq = 8
    h_freq = 30
    
    ovr_labels = preprocessing.create_ovr_dataset(labels)

    classifiers = []
    ovr_spatial_filters = []

    mibif_indices = []

    if mode == 'MIBIF':
        n_freq_bands = len(trials)
        n_chans_csp = 2*n_csp
        mibif_pairs_dictionary = MIBIF.get_mibif_pairs_dictionary(n_freq_bands,n_chans_csp)
        
    for binary_labels in ovr_labels:

        if mode == 'MIBIF':
            cur_classif_features,cur_classif_labels,cur_classif_spatial_filter,cur_mibif_indices= handle_feature_extraction(trials,
                                                                                                                            binary_labels,
                                                                                                                            mode,
                                                                                                                            mibif_pairs_dictionary = mibif_pairs_dictionary)
                
            mibif_indices.append(cur_mibif_indices)

    
        else:
            cur_classif_features,cur_classif_labels,cur_classif_spatial_filter = handle_feature_extraction(trials,binary_labels,mode = mode)
        
        ovr_spatial_filters.append(cur_classif_spatial_filter)
        
        #To prevent overwriting of individual classifiers
        base_classifier_copy = copy.copy(base_classifier)
        base_classifier_copy.fit(cur_classif_features,cur_classif_labels)
        
        classifiers.append(base_classifier_copy)

    if mode == 'MIBIF':
        return classifiers,ovr_spatial_filters,mibif_indices

    return classifiers,ovr_spatial_filters

def handle_feature_extraction(trials,binary_labels,mode,n_csp = 2,n_mibif_pairs = 4,mibif_pairs_dictionary = None):

    return feature_extraction.handle_feature_extraction(trials,binary_labels,mode,n_csp = n_csp,n_mibif_pairs = n_mibif_pairs,mibif_pairs_dictionary = mibif_pairs_dictionary)

    


def make_predictions(trials,ovr_classifiers,ovr_spatial_filters,mode = 'CSP',mibif_indices = None):

    if mode == 'CSP':
        ovr_features = np.array([CSP.get_log_var_csp_features(trials,W) for W in ovr_spatial_filters])
    elif mode == 'FBCSP':
        ovr_features = np.array([CSP.get_log_var_fbank_csp_features(trials,W) for W in ovr_spatial_filters])

    elif mode == 'MIBIF':
        ovr_features = np.array([CSP.get_log_var_fbank_csp_features(trials,W) for W in ovr_spatial_filters])
    
        mibif_selected_features = []

        #Numpy does not like ragged tensors, so I have to do it in this kind of slow way unlike the above ones with comprehensions
        for i,indices in enumerate(mibif_indices):
            cur_mibif_features = ovr_features[i][:,indices]
            mibif_selected_features.append(cur_mibif_features)

        ovr_features = mibif_selected_features

    
    ovr_probas = np.array([ovr_classifiers[i].predict_proba(ovr_features[i]) for i in range(len(ovr_classifiers))]).transpose(1,0,2)

    multi_label_predictions = np.argmax(ovr_probas[:,:,0],axis = 1)

    return multi_label_predictions





    







    
    
        












    