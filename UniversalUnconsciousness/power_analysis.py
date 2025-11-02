import h5py
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm
from .hdf5_utils import convert_h5_string_array, TransposedDatasetView

freq_bands = {
    'delta': [0.1, 4],
    'theta': [4, 8],
    'alpha': [8, 12],
    'beta': [12, 30],
    'gamma': [30, 100]
}

def perform_fft_analysis(delase_results, cfg, session, area, verbose=False):
    if 'propofol' in cfg.params.data_class:
        session_file = h5py.File(os.path.join(cfg.params.all_data_dir, 'anesthesia', 'mat', cfg.params.data_class, f'{session}.mat'), 'r')
    else:
        session_file = h5py.File(os.path.join(cfg.params.all_data_dir, cfg.params.data_class, 'mat', f'{session}.mat'), 'r')
    
    # electrode_areas = convert_h5_string_array(session_file, session_file['electrodeInfo']['area'])
    # area_inds = {area: np.where(electrode_areas == area)[0] if area != 'all' else np.arange(len(electrode_areas)) for area in areas}

    lfp = TransposedDatasetView(session_file['lfp']).transpose()
    dt = session_file['lfpSchema']['smpInterval'][0, 0]
    window = int((delase_results[session][area].window_end.iloc[0] - delase_results[session][area].window_start.iloc[0])/dt)
    freqs = np.fft.rfftfreq(window, d=dt)
    freq_powers = {freq_band: np.zeros(len(delase_results[session][area])) for freq_band in freq_bands.keys()}
    ffts = []
    for i, row in tqdm(delase_results[session][area].iterrows(), total=len(delase_results[session][area]), disable=not verbose):
        lfp_window = lfp[int(row.window_start/dt):int(row.window_end/dt)]
        try:
            fft_time = np.fft.rfft(lfp_window, axis=0) # FFT along time dimension
        except ValueError as e:
            print(f"Error in FFT for {session} {area} {i}: {e}")
            print(i, row.window_start, row.window_end)
            print(lfp_window.shape)
            raise e
            
        ffts.append(fft_time)
    ffts = np.stack(ffts)
    return ffts, freqs

def perform_power_analysis(delase_results, cfg, session, area, top_percent=0.1, verbose=False):
    if 'propofol' in cfg.params.data_class:
        session_file = h5py.File(os.path.join(cfg.params.all_data_dir, 'anesthesia', 'mat', cfg.params.data_class, f'{session}.mat'), 'r')
    else:
        session_file = h5py.File(os.path.join(cfg.params.all_data_dir, cfg.params.data_class, 'mat', f'{session}.mat'), 'r')
    
    # electrode_areas = convert_h5_string_array(session_file, session_file['electrodeInfo']['area'])
    # area_inds = {area: np.where(electrode_areas == area)[0] if area != 'all' else np.arange(len(electrode_areas)) for area in areas}

    lfp = TransposedDatasetView(session_file['lfp']).transpose()
    dt = session_file['lfpSchema']['smpInterval'][0, 0]
    window = int((delase_results[session][area].window_end.iloc[0] - delase_results[session][area].window_start.iloc[0])/dt)
    freqs = np.fft.rfftfreq(window, d=dt)
    freq_powers = {freq_band: np.zeros(len(delase_results[session][area])) for freq_band in freq_bands.keys()}
    for i, row in tqdm(delase_results[session][area].iterrows(), total=len(delase_results[session][area]), disable=not verbose):
        lfp_window = lfp[int(row.window_start/dt):int(row.window_end/dt)]
        fft_time = np.fft.rfft(lfp_window, axis=0) # FFT along time dimension
        for freq_band, freq_bounds in freq_bands.items():
            freq_mask = (freqs >= freq_bounds[0]) & (freqs <= freq_bounds[1])
            fft_power = np.abs(fft_time[freq_mask])**2
            # take the mean over electrodes and sum over frequencies
            freq_powers[freq_band][i] += fft_power.mean(axis=1).sum(axis=0) # freqs x channels
    
    # print the r-squared value
    stab_means = delase_results[session][area].stability_params.apply(lambda x: x[:int(top_percent*len(x))].mean())
    freq_r2_scores = {}
    for freq_band, power_series in freq_powers.items():
        reg = LinearRegression().fit(power_series.reshape(-1, 1), stab_means.values)
        freq_r2_scores[freq_band] = reg.score(power_series.reshape(-1, 1), stab_means.values)
    return freq_powers, freq_r2_scores