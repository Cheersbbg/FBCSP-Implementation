import numpy as np
from sklearn.feature_selection import mutual_info_classif

### A lot of implementations of FBCSP make the mistake
# of getting this pairs dictionary over the entire feature vector,
# not considering which spatial filter generated which elements of the feature vector
def get_mibif_pairs_dictionary(n_freq_bands,n_chans_csp):
    
    mibif_template = np.array([0,1,2,3])
    
    mibif_pairs_dict = {}
    
    for i in range(n_freq_bands):
        cur_feature_indices = mibif_template + (n_chans_csp*i)
        associated_pairs = reversed(cur_feature_indices)
    
        cur_mibif_pairs = dict(zip(cur_feature_indices,associated_pairs))
    
        mibif_pairs_dict.update(cur_mibif_pairs)

    return mibif_pairs_dict


def select_mibif_features(initial_feature_vectors,labels,mibif_pairs_dictionary,n_mibif_pairs = 4):

    mi_scores = mutual_info_classif(initial_feature_vectors,labels)
    sorted_idxs = np.argsort(mi_scores)[::-1]

    selected_indices = []

    for idx in sorted_idxs[:n_mibif_pairs]:

        selected_indices.append(idx)
        associated_pair = mibif_pairs_dictionary[idx]

        if associated_pair in sorted_idxs[:n_mibif_pairs]:
            continue
        selected_indices.append(associated_pair)

    return selected_indices
