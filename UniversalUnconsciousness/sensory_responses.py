from delase import embed_signal_torch
import h5py
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import statsmodels.tsa.stattools as smt
import torch
from tqdm.auto import tqdm
import pandas as pd
from .data_utils import get_section_info
from .hdf5_utils import convert_h5_string_array, TransposedDatasetView


def get_sensory_responses_leverOddball(cfg, session, noise_filter_info, trial_type = 'lvr_tone', leadup = 250, response = 250):
    section_info, section_info_extended, section_colors, infusion_start = get_section_info(session, cfg.params.all_data_dir, cfg.params.data_class)
    section_info_dict = {name: times for name, times in section_info}
    
    session_file = h5py.File(os.path.join(cfg.params.all_data_dir, cfg.params.data_class, 'mat', f"{session}.mat"))

    # ------------------------------
    # Get trial variables
    # ------------------------------
    if trial_type == 'oddball':
        # task_types = convert_h5_string_array(session_file, session_file['trialInfo']['task'])
        # trial_starts = session_file['trialInfo']['trialStart'][:, 0]
        # trial_ends = session_file['trialInfo']['trialEnd'][:, 0]

        # odd_tone_onsets = session_file['trialInfo']['odd_toneOnsets'][:, 0]
        # odd_tone_offsets = session_file['trialInfo']['odd_toneOffsets'][:, 0]

        odd_sequence_start = session_file['trialInfo']['odd_sequenceStart'][:, 0]
        odd_sequence_end = session_file['trialInfo']['odd_sequenceEnd'][:, 0]

        # oddball_trial_starts = trial_starts[task_types == 'oddball']
        # oddball_trial_ends = trial_ends[task_types == 'oddball']

        # odd_sequence = convert_h5_string_array(session_file, session_file['trialInfo']['odd_sequence'])
        # unique_sequences = np.unique(odd_sequence)

        start_times = odd_sequence_start
        end_times = odd_sequence_end

    elif trial_type == 'lvr_tone':
        lvr_tone_onsets = session_file['trialInfo']['lvr_toneOnset'][:, 0]
        lvr_tone_offsets = session_file['trialInfo']['lvr_toneOffset'][:, 0]

        start_times = lvr_tone_onsets
        end_times = lvr_tone_offsets
    else:
        raise ValueError(f"Trial type {trial_type} not supported")


    lfp = TransposedDatasetView(session_file['lfp']).transpose()
    dt = session_file['lfpSchema']['smpInterval'][0, 0]

    valid_electrodes = np.arange(lfp.shape[1])[~np.isin(np.arange(lfp.shape[1]), noise_filter_info[session]['bad_electrodes'])]

    valid_window_starts = noise_filter_info[session]['valid_window_starts']
    valid_window_ends = valid_window_starts + cfg.params.window

    # ------------------------------
    # Lever tone responses
    # ------------------------------

    # First pass - count number of valid tones per section
    
    if trial_type == 'lvr_tone':
        awake_section = 'awake lever2'
        anesthesia_section = 'early unconscious'
        recovery_section = 'late unconscious'
    else:
        awake_section = 'awake oddball'
        anesthesia_section = 'unconscious oddball'
        recovery_section = 'recovery oddball'
    section_tone_counts = {
        awake_section: 0,
        anesthesia_section: 0,
        recovery_section: 0
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

def get_sensory_responses_propofol(cfg, session, noise_filter_info, trial_type, leadup = 250, response = 250, area='all'):
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

    areas = convert_h5_string_array(session_file, session_file['electrodeInfo']['area'])
    if area == 'all':
        electrode_inds = np.arange(lfp.shape[1])
    else:
        electrode_inds = np.where(areas == area)[0]

    valid_electrodes = np.arange(lfp.shape[1])[~np.isin(electrode_inds, noise_filter_info[session]['bad_electrodes'])]

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
        "maintenance dose": 0,
        "recovery": 0
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
        n_delays = 1,
        delay_interval = 1,
        use_mean = False
    ):
    
    responses_etdc = {}

    for monkey in sensory_responses.keys():
        responses_etdc[monkey] = {}
        for dose in sensory_responses[monkey].keys():
            responses_etdc[monkey][dose] = {
                'awake': [],
                'recovery': [],
                'unconscious': []
            }
            for session in sensory_responses[monkey][dose].keys():
                for section in sensory_responses[monkey][dose][session]:
                    responses = sensory_responses[monkey][dose][session][section]
                    responses_de = embed_signal_torch(responses, n_delays, delay_interval)
                    if len(responses_de) == 0:
                        continue
                    if use_mean:
                        temp = np.expand_dims(responses_de.mean(axis=(0, 2)), axis=-1)
                    else:
                        print(responses_de.shape)
                        pca = PCA(n_components=2).fit(responses_de.mean(axis=0))
                        temp = pca.transform(responses_de.mean(axis=0))
                        # pca = PCA(n_components=2).fit(responses_de.reshape(-1, responses_de.shape[-1]))
                        # temp = pca.transform(responses_de.reshape(-1, responses_de.shape[-1])).reshape(responses_de.shape[0], responses_de.shape[1], 2)
                        # temp = temp.mean(axis=0)
                    # print(temp.shape)
                    if 'awake' in section:
                        responses_etdc[monkey][dose]['awake'].append(temp)
                    elif 'recovery' in section or 'late unconscious' in section:
                        responses_etdc[monkey][dose]['recovery'].append(temp)
                    else:
                        responses_etdc[monkey][dose]['unconscious'].append(temp)
                    
            responses_etdc[monkey][dose]['awake'] = np.array(responses_etdc[monkey][dose]['awake'])
            responses_etdc[monkey][dose]['recovery'] = np.array(responses_etdc[monkey][dose]['recovery']) if len(responses_etdc[monkey][dose]['recovery']) > 0 else None
            responses_etdc[monkey][dose]['unconscious'] = np.array(responses_etdc[monkey][dose]['unconscious'])
    return responses_etdc

def get_responses_acf(
        sensory_responses,
        agent,
        response,
        n_delays = 1,
        delay_interval = 1,
        method = 'grouped',
        use_mean = False,
        n_lags = 50,
        n_ac_pts = None,
        verbose = False,
        data_save_dir = None
    ):
    '''
    method:
        'grouped': compute the autocorrelation for the mean of the responses
        'individual': compute the autocorrelation for each response (across all electrodes), and then average across electrodes
        'cosine': compute the autocorrelation as high-dimensional cosine similarity of the responses (across all electrodes)
    '''

    filename = f'{agent}_{response}_sensory_responses_acf_{n_delays}_{delay_interval}_{method}_{use_mean}_{n_lags}_{n_ac_pts}.pkl'
    if data_save_dir is not None:
        if os.path.exists(os.path.join(data_save_dir, filename)):
            return pd.read_pickle(os.path.join(data_save_dir, filename))

    responses_autocorrelation = {}

    n_iterations = 0
    for monkey in sensory_responses.keys():
        for dose in sensory_responses[monkey].keys():
            for session in sensory_responses[monkey][dose].keys():
                for section in sensory_responses[monkey][dose][session].keys():
                    n_iterations += 1

    iterator = tqdm(total=n_iterations, desc='Computing ACF', disable=not verbose)

    for monkey in sensory_responses.keys():
        responses_autocorrelation[monkey] = {}
        for dose in sensory_responses[monkey].keys():
            responses_autocorrelation[monkey][dose] = {
                'awake': [],
                'unconscious': [],
                'recovery': [],
            }
            for session in sensory_responses[monkey][dose].keys():
                for section in sensory_responses[monkey][dose][session]:
                    responses = sensory_responses[monkey][dose][session][section]
                    responses_de = embed_signal_torch(responses, n_delays, delay_interval)
                    if n_ac_pts is None:
                        n_ac_pts = responses_de.shape[1]
                    responses_de = responses_de[:, :n_ac_pts]
                    if len(responses_de) == 0:
                        continue
                    if method == 'grouped':
                        if use_mean:
                            temp = np.expand_dims(responses_de.mean(axis=(0, 2)), axis=-1)[:, 0]
                        else:
                            pca = PCA(n_components=1).fit(responses_de.mean(axis=0))
                            temp = pca.transform(responses_de.mean(axis=0))[:, 0]
                        # temp is (time, 1)
                        autocorrelation = smt.acf(temp, nlags=n_lags)
                    elif method == 'individual':
                        # trials x time x electrodes
                        autocorrelation = np.array([smt.acf(responses_de[i, :, j], nlags=n_lags) for i in range(responses_de.shape[0]) for j in range(responses_de.shape[2])])
                        # (trials * electrodes) x autocorrelation time (n_lags + 1)
                        autocorrelation = autocorrelation.mean(axis=0)
                    elif method == 'cosine':
                        corrmat = cosine_sim_corrmat(responses_de)
                        autocorrelation = np.array([np.mean([np.diag(corrmat[i], k=k).flatten() for i in range(corrmat.shape[0])]) for k in range(n_lags + 1)])
                    else:
                        raise ValueError(f"Method {method} not supported")
                    if 'awake' in section:
                        responses_autocorrelation[monkey][dose]['awake'].append(autocorrelation)
                    elif 'recovery' in section or 'late unconscious' in section:
                        responses_autocorrelation[monkey][dose]['recovery'].append(autocorrelation)
                    else:
                        responses_autocorrelation[monkey][dose]['unconscious'].append(autocorrelation)
                    iterator.update(1)
            responses_autocorrelation[monkey][dose]['awake'] = np.array(responses_autocorrelation[monkey][dose]['awake'])
            responses_autocorrelation[monkey][dose]['recovery'] = np.array(responses_autocorrelation[monkey][dose]['recovery']) if len(responses_autocorrelation[monkey][dose]['recovery']) > 0 else None
            responses_autocorrelation[monkey][dose]['unconscious'] = np.array(responses_autocorrelation[monkey][dose]['unconscious'])
    iterator.close()
    if data_save_dir is not None:
        pd.to_pickle(responses_autocorrelation, os.path.join(data_save_dir, filename))
    return responses_autocorrelation


def cosine_sim_corrmat(responses_vecs):
    responses_vecs = responses_vecs - np.expand_dims(responses_vecs.mean(axis=1), axis=1)
    cosine_sim_corrmats = np.zeros((responses_vecs.shape[-3], responses_vecs.shape[-2], responses_vecs.shape[-2]))
    for trial_num in range(responses_vecs.shape[-3]):
        cosine_sim_corrmats[trial_num] = cosine_similarity(responses_vecs[trial_num])
    return cosine_sim_corrmats
