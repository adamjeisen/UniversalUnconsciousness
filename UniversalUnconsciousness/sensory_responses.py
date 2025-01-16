from delase import embed_signal_torch
import h5py
import numpy as np
import os
from sklearn.decomposition import PCA

from .data_utils import get_section_info
from .hdf5_utils import convert_h5_string_array, TransposedDatasetView


def get_sensory_responses_leverOddball(cfg, session, noise_filter_info, leadup = 250, response = 250):
    section_info, section_info_extended, section_colors, infusion_start = get_section_info(session, cfg.params.all_data_dir, cfg.params.data_class)
    section_info_dict = {name: times for name, times in section_info}
    
    session_file = h5py.File(os.path.join(cfg.params.all_data_dir, cfg.params.data_class, 'mat', f"{session}.mat"))

    # ------------------------------
    # Get trial variables
    # ------------------------------
    task_types = convert_h5_string_array(session_file, session_file['trialInfo']['task'])
    trial_starts = session_file['trialInfo']['trialStart'][:, 0]
    trial_ends = session_file['trialInfo']['trialEnd'][:, 0]

    # odd_tone_onsets = session_file['trialInfo']['odd_toneOnsets'][:, 0]
    # odd_tone_offsets = session_file['trialInfo']['odd_toneOffsets'][:, 0]

    # odd_sequence_start = session_file['trialInfo']['odd_sequenceStart'][:, 0]
    # odd_sequence_end = session_file['trialInfo']['odd_sequenceEnd'][:, 0]

    # oddball_trial_starts = trial_starts[task_types == 'oddball']
    # oddball_trial_ends = trial_ends[task_types == 'oddball']

    # odd_sequence = convert_h5_string_array(session_file, session_file['trialInfo']['odd_sequence'])
    # unique_sequences = np.unique(odd_sequence)

    lvr_tone_onsets = session_file['trialInfo']['lvr_toneOnset'][:, 0]
    lvr_tone_offsets = session_file['trialInfo']['lvr_toneOffset'][:, 0]

    # lvr_outcome = convert_h5_string_array(session_file, session_file['trialInfo']['lvr_outcome'])
    # binary_outcome = [outcome == 'correct' for outcome in lvr_outcome]

    lfp = TransposedDatasetView(session_file['lfp']).transpose()
    dt = session_file['lfpSchema']['smpInterval'][0, 0]

    valid_electrodes = np.arange(lfp.shape[1])[~np.isin(np.arange(lfp.shape[1]), noise_filter_info[session]['bad_electrodes'])]

    valid_window_starts = noise_filter_info[session]['valid_window_starts']
    valid_window_ends = valid_window_starts + cfg.params.window

    # ------------------------------
    # Lever tone responses
    # ------------------------------
    start_times = lvr_tone_onsets
    end_times = lvr_tone_offsets

    # First pass - count number of valid tones per section
    section_tone_counts = {
        "awake lever2": 0,
        "early unconscious": 0
    }

    # Check which tones are valid and count them per section
    for start_time, end_time in zip(start_times, end_times):
        # Check if tone is in a valid recording window
        in_valid_window = False
        for valid_start, valid_end in zip(valid_window_starts, valid_window_ends):
            if start_time >= valid_start and end_time <= valid_end:
                in_valid_window = True
                break
        
        if in_valid_window:
            # Check which section this tone belongs to
            for section, times in section_info_dict.items():
                if section in section_tone_counts.keys():
                    if start_time >= times[0]*60 + infusion_start and end_time <= times[-1]*60 + infusion_start:
                        section_tone_counts[section] += 1

    # Initialize arrays with correct sizes
    tone_lfps = {
        section: np.empty((count, leadup + response, len(valid_electrodes)))
        for section, count in section_tone_counts.items()
    }

    # Reset counters for filling arrays
    tone_counts = {section: 0 for section in section_tone_counts.keys()}

    # Second pass - fill arrays
    for start_time, end_time in zip(start_times, end_times):
        # Check if tone is in a valid recording window
        in_valid_window = False
        for valid_start, valid_end in zip(valid_window_starts, valid_window_ends):
            if start_time >= valid_start and end_time <= valid_end:
                in_valid_window = True
                break
                
        if in_valid_window:
            for section, times in section_info_dict.items():
                if section in tone_lfps:
                    if start_time >= times[0]*60 + infusion_start and end_time <= times[-1]*60 + infusion_start:
                        index = tone_counts[section]
                        tone_lfps[section][index] = lfp[int(start_time/dt) - leadup:int(start_time/dt) + response, valid_electrodes]
                        tone_counts[section] += 1

    return tone_lfps, dt

def get_sensory_responses_propofol(cfg, session, noise_filter_info, trial_type, leadup = 250, response = 250):
    section_info, section_info_extended, section_colors, infusion_start = get_section_info(session, cfg.params.all_data_dir, cfg.params.data_class)
    section_info_dict = {name: times for name, times in section_info}
    
    session_file = h5py.File(os.path.join(cfg.params.all_data_dir, 'anesthesia', 'mat', cfg.params.data_class, f"{session}.mat"))

    trial_types = convert_h5_string_array(session_file, session_file['trialInfo']['cpt_trialType'])
    if trial_type == 'toneOnly':
        trial_starts = session_file['trialInfo']['cpt_toneOn'][:, 0]
        trial_ends = session_file['trialInfo']['cpt_toneOff'][:, 0]
    elif trial_type == 'tonePuff':
        trial_starts = session_file['trialInfo']['cpt_toneOn'][:, 0]
        trial_ends = session_file['trialInfo']['cpt_toneOff'][:, 0]

    trial_starts = trial_starts[trial_types == trial_type]
    trial_ends = trial_ends[trial_types == trial_type]

    lfp = TransposedDatasetView(session_file['lfp']).transpose()
    dt = session_file['lfpSchema']['smpInterval'][0, 0]

    valid_electrodes = np.arange(lfp.shape[1])[~np.isin(np.arange(lfp.shape[1]), noise_filter_info[session]['bad_electrodes'])]

    valid_window_starts = noise_filter_info[session]['valid_window_starts']
    valid_window_ends = valid_window_starts + cfg.params.window

    # ------------------------------
    # Sensory responses
    # ------------------------------
    start_times = trial_starts
    end_times = trial_ends

    # First pass - count number of valid tones per section
    section_sensory_counts = {
        "awake": 0,
        "maintenance dose": 0
    }

    # Check which tones are valid and count them per section
    for start_time, end_time in zip(start_times, end_times):
        # Check if tone is in a valid recording window
        in_valid_window = False
        for valid_start, valid_end in zip(valid_window_starts, valid_window_ends):
            if start_time >= valid_start and end_time <= valid_end:
                in_valid_window = True
                break
        
        if in_valid_window:
            # Check which section this tone belongs to
            for section, times in section_info_dict.items():
                if section in section_sensory_counts.keys():
                    if start_time >= times[0]*60 + infusion_start and end_time <= times[-1]*60 + infusion_start:
                        section_sensory_counts[section] += 1

    # Initialize arrays with correct sizes
    sensory_responses = {
        section: np.empty((count, leadup + response, len(valid_electrodes)))
        for section, count in section_sensory_counts.items()
    }

    # Reset counters for filling arrays
    sensory_counts = {section: 0 for section in section_sensory_counts.keys()}

    # Second pass - fill arrays
    for start_time, end_time in zip(start_times, end_times):
        # Check if tone is in a valid recording window
        in_valid_window = False
        for valid_start, valid_end in zip(valid_window_starts, valid_window_ends):
            if start_time >= valid_start and end_time <= valid_end:
                in_valid_window = True
                break
                
        if in_valid_window:
            for section, times in section_info_dict.items():
                if section in sensory_responses:
                    if start_time >= times[0]*60 + infusion_start and end_time <= times[-1]*60 + infusion_start:
                        index = sensory_counts[section]
                        sensory_responses[section][index] = lfp[int(start_time/dt) - leadup:int(start_time/dt) + response, valid_electrodes]
                        sensory_counts[section] += 1

    return sensory_responses, dt

def get_responses_etdc(
        sensory_responses,
        agent,  
        leadup,
        n_delays = 1,
        delay_interval = 1
    ):
    
    responses_etdc = {}

    for monkey in sensory_responses.keys():
        responses_etdc[monkey] = {}
        for dose in sensory_responses[monkey].keys():
            responses_etdc[monkey][dose] = {
                'awake': [],
                'unconscious': []
            }
            for session in sensory_responses[monkey][dose].keys():
                for section in sensory_responses[monkey][dose][session]:
                    responses = sensory_responses[monkey][dose][session][section]
                    responses_de = embed_signal_torch(responses, n_delays, delay_interval)
                    if len(responses_de) == 0:
                        continue
                    pca = PCA(n_components=2).fit(responses_de.mean(axis=0))
                    temp = pca.transform(responses_de.mean(axis=0))
                    # # ensure responses deflect downwards first
                    # for j in range(temp.shape[-1]):
                    #     if agent == 'propofol':
                    #         first_portion = temp[leadup:leadup+100, j]
                    #     else:
                    #         first_portion = temp[leadup:leadup+50, j]
                    #     # perform a linear regression on the first 100 ms
                    #     slope, intercept = np.polyfit(np.arange(first_portion.shape[0]), first_portion, 1)
                    #     if slope > 0:
                    #         temp[:, j] = -temp[:, j]
                    if 'awake' in section:
                        responses_etdc[monkey][dose]['awake'].append(temp)
                    else:
                        responses_etdc[monkey][dose]['unconscious'].append(temp)
            responses_etdc[monkey][dose]['awake'] = np.array(responses_etdc[monkey][dose]['awake'])
            responses_etdc[monkey][dose]['unconscious'] = np.array(responses_etdc[monkey][dose]['unconscious'])
    return responses_etdc
