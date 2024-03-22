from src import MIBIF,preprocessing,CSP


def handle_feature_extraction(trials,labels,mode,n_csp = 2,n_mibif_pairs = 4,mibif_pairs_dictionary = None):

    preprocessing.shuffle_trials_and_labels(trials,labels)
    
    if mode == 'CSP':

        classif_spatial_filters = CSP.get_spatial_filters(trials,labels, n_csp = n_csp)
        classif_features = CSP.get_log_var_csp_features(trials,classif_spatial_filters)


    elif mode == 'FBCSP':
        classif_spatial_filters = CSP.get_filter_bank_spatial_filters(trials,labels)
        classif_features = CSP.get_log_var_fbank_csp_features(trials,classif_spatial_filters)

    elif mode == 'MIBIF':
        classif_spatial_filters = CSP.get_filter_bank_spatial_filters(trials,labels)
        classif_features = CSP.get_log_var_fbank_csp_features(trials,classif_spatial_filters)

        mibif_indices = MIBIF.select_mibif_features(classif_features,labels,mibif_pairs_dictionary,n_mibif_pairs = n_mibif_pairs)
        classif_features = classif_features[:,mibif_indices]

    if mode == 'MIBIF':
        return classif_features,labels,classif_spatial_filters,mibif_indices

    return classif_features,labels,classif_spatial_filters