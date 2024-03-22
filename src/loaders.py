import mne
mne.set_log_level('ERROR')

import numpy as np
from scipy.io import loadmat

# By default loading the trials belonging to training data. Set training = False if you would like to load the evaluation data
def load_events(subject_no = None,training = True,recording = None):
    if training:
        recording_path = f'GDF Files/A0{subject_no}T.gdf'
    else:
        recording_path = f'GDF Files/A0{subject_no}E.gdf'

    if not recording:
        recording = mne.io.read_raw_gdf(recording_path,preload = True)

    events,events_dict = mne.events_from_annotations(recording)

    #Excluding the EOG Channels and scaling the data to not be in microvolts
    return events,events_dict,recording.get_data()[:-3]*10**6


def get_info_dictionary(subject_no,return_recording = False,training = True):
    if training:
        recording_path = f'GDF Files/A0{subject_no}T.gdf'
    else:
        recording_path = f'GDF Files/A0{subject_no}E.gdf'

    recording = mne.io.read_raw_gdf(recording_path,preload = True)

    ch_names = ['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']
    sfreq = recording.info['sfreq']

    ch_types = ['eeg']*len(ch_names)

    altered_info_structure = mne.create_info(ch_names = ch_names,ch_types = ch_types, sfreq = sfreq)

    if return_recording:
        return recording,altered_info_structure

    return altered_info_structure
    


# By default we are not including the artifactual/rejected trials. For every event id belonging
# to one of the classes of data, if the next element has an event id of 1, then it is indicating that the trial is artifactual, and 
# we do not include it

# By default as done in the FBCSP paper, we start by extracting in a radius of 4.5 seconds from the onset of the cue of a motor imagery trial.
# This is just for the training files since the evaluation data uses different codes for events so as to not reveal the competition labels, the
# competition labels themselves are stored elsewhere
def extract_motor_imagery_trials(subject_no = None,fs = 250,training = True,t_start = -4.5,t_end = 4.5,recording = None):

    if not recording:
        events,events_dict,recording = load_events(subject_no,training = training)
    else:
        events,events_dict,recording = load_events(recording = recording)

    class_related_event_ids = np.array([7,8,9,10])
    
    #subject 4 has a weird GDF file that's different than the rest, so have to handle for that here 
    if subject_no == 4:
        class_related_event_ids = [5,6,7,8]

    min_event_id = min(class_related_event_ids)
        
    trials = []
    labels = []
    
    for i,event in enumerate(events):

        event_cue_onset,event_id = event[0],event[2]

        is_artifactual = events[:,2][i-1] == 1
        
        if is_artifactual:continue

        elif event_id in class_related_event_ids:

            trial_extraction_start = int(event_cue_onset + t_start*fs)
            trial_extraction_end = int(event_cue_onset + t_end*fs)

            extracted_trial = recording[:,trial_extraction_start:trial_extraction_end]
            
            trials.append(extracted_trial)
            labels.append(event_id - min_event_id)

    return np.array(trials),np.array(labels)



def load_evaluation_trials(subject_no,data_dir,labels_dir,fs = 250,t_start = -4.5,t_end = 4.5):
    events,events_dict,recording = load_events(subject_no,training = False)

    evaluation_trials = []
    
    for i,event in enumerate(events):
        event_cue_onset,event_id = event[0],event[2]

        
        is_artifactual = events[:,2][i-1] == 1
        
        if event_id == 7:
            
            trial_extraction_start = int(event_cue_onset + t_start*fs)
            trial_extraction_end = int(event_cue_onset + t_end*fs)

            extracted_trial = recording[:,trial_extraction_start:trial_extraction_end]
            
            evaluation_trials.append(extracted_trial)

    evaluation_labels = loadmat(f'Evaluation Labels/A0{subject_no}E.mat')['classlabel'][:,0] - 1

    return np.array(evaluation_trials),evaluation_labels
    

            











        
