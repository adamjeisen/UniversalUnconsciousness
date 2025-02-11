from copy import deepcopy
import h5py
import numpy as np
import os
import pandas as pd
import scipy.signal as signal
from sklearn.decomposition import PCA
# from spynal.matIO import loadmat
import time
from tqdm.auto import tqdm

from .hdf5_utils import *

def get_data_class(session, all_data_dir):
    data_class = None
    for (dirpath, dirnames, filenames) in os.walk(all_data_dir):
        if f"{session}.mat" in filenames:
            data_class = os.path.basename(dirpath)
            break
    if data_class is None:
        raise ValueError(f"Neural data for session {session} could not be found in the provided folder.")

    return data_class

def load_session_data(session, all_data_dir, variables, data_class=None, verbose=True):   
    if data_class is None:
        data_class = get_data_class(session, all_data_dir)
    
    filename = os.path.join(all_data_dir, data_class, 'mat', f'{session}.mat')

    start = time.process_time()
    if 'lfpSchema' not in variables:
        variables.append('lfpSchema')

    if verbose:
        print(f"Loading data: {variables}...")
    start = time.process_time()
    session_vars = {}
    for arg in variables:
        session_vars[arg] = loadmat(filename, variables=[arg], verbose=verbose)
    if verbose:
        print(f"Data loaded (took {time.process_time() - start:.2f} seconds)")

    if 'electrodeInfo' in variables:
        if session in ['MrJones-Anesthesia-20160201-01', 'MrJones-Anesthesia-20160206-01', 'MrJones-Anesthesia-20160210-01']:
            session_vars['electrodeInfo']['area'] = np.delete(session_vars['electrodeInfo']['area'], np.where(np.arange(len(session_vars['electrodeInfo']['area'])) == 60))
            session_vars['electrodeInfo']['channel'] = np.delete(session_vars['electrodeInfo']['channel'], np.where(np.arange(len(session_vars['electrodeInfo']['channel'])) == 60))
            session_vars['electrodeInfo']['NSP'] = np.delete(session_vars['electrodeInfo']['NSP'], np.where(np.arange(len(session_vars['electrodeInfo']['NSP'])) == 60))
        # elif data_class == 'leverOddball':
        if 'leverOddball' in filename or 'LvrOdd' in filename:
            session_vars['electrodeInfo']['area'] = np.array([f"{area}-{h[0].upper()}" for area, h in zip(session_vars['electrodeInfo']['area'], session_vars['electrodeInfo']['hemisphere'])])
    if session in ['PEDRI_Ketamine_20220113']:
        if 'sessionInfo' in session_vars.keys():
            session_vars['sessionInfo']['infusionStart'] = session_vars['sessionInfo']['infusionStart'][-1]
    T = len(session_vars['lfpSchema']['index'][0])
    N = len(session_vars['lfpSchema']['index'][1])
    dt = session_vars['lfpSchema']['smpInterval'][0]

    return session_vars, T, N, dt

def save_lfp_chunks(session, all_data_dir="/scratch2/weka/millerlab/eisenaj/datasets/anesthesia/mat", chunk_time_s=20):
    data_class = get_data_class(session, all_data_dir)
    
    filename = os.path.join(all_data_dir, data_class, f'{session}.mat')
    print("Loading data ...")
    start = time.process_time()
    lfp, lfp_schema = loadmat(filename, variables=['lfp', 'lfpSchema'], verbose=False)
    dt = lfp_schema['smpInterval'][0]
    fs = 1/dt
    print(f"Data loaded (took {time.process_time() - start:.2f} seconds)")
    
    save_dir = os.path.join(all_data_dir, data_class, f"{session}_lfp_chunked_{chunk_time_s}s")
    os.makedirs(save_dir, exist_ok=True)
    
    chunk_width = int(chunk_time_s*fs)
    num_chunks = int(np.ceil(lfp.shape[0]/chunk_width))
    directory = []
    for i in tqdm(range(num_chunks)):
        start_ind = i*chunk_width
        end_ind = np.min([(i+1)*chunk_width, lfp.shape[0]])
        chunk = lfp[start_ind:end_ind]
        filepath = os.path.join(save_dir, f"chunk_{i}")
        if os.path.exists(filepath):
            print(f"Chunk at {filepath} already exists")
        else:
            pd.to_pickle(chunk, filepath)
            directory.append(dict(
                start_ind=start_ind,
                end_ind=end_ind,
                filepath=filepath,
                start_time=start_ind*dt,
                end_time=end_ind*dt
            ))
    
    directory = pd.DataFrame(directory)
    
    pd.to_pickle(directory, os.path.join(save_dir, "directory"))
#         print(f"Chunk: {start_ind/(1000*60)} min to {end_ind/(1000*60)} 

def load_window_from_chunks(window_start, window_end, directory, dimension_inds=None):
    dt = directory.end_time.iloc[0]/directory.end_ind.iloc[0]
    fs = 1/dt
    window_start = int(window_start*fs)
    window_end = int(window_end*fs)
    
    start_time_bool = directory.start_ind <= window_start
    start_row = np.argmin(start_time_bool) - 1 if np.sum(start_time_bool) < len(directory) else len(directory) - 1
    end_time_bool = directory.end_ind > window_end
    end_row = np.argmax(end_time_bool) if np.sum(end_time_bool) > 0 else len(directory) - 1
    
    window_data = None
    
    pos_in_window = 0
    for row_ind in range(start_row, end_row + 1):
        row = directory.iloc[row_ind]
        chunk = pd.read_pickle(row.filepath)
        if dimension_inds is None:
            dimension_inds = np.arange(chunk.shape[1])
        if window_data is None:
            window_data = np.zeros((window_end - window_start, len(dimension_inds)))
                
        if row.start_ind <= window_start:
            start_in_chunk = window_start - row.start_ind
        else:
            start_in_chunk = 0

        if row.end_ind <= window_end:
            end_in_chunk = chunk.shape[0]
        else:
            end_in_chunk = window_end - row.start_ind

        window_data[pos_in_window:pos_in_window + end_in_chunk - start_in_chunk] = chunk[start_in_chunk:end_in_chunk, dimension_inds]
        pos_in_window += end_in_chunk - start_in_chunk
                
    return window_data

# grid stuff

def compile_grid_results(session, grid_search_results_dir, areas=None, normed=False):
    if normed is False:
        norm_folder = 'NOT_NORMED'
    else:
        norm_folder = 'NORMED'

    if areas is None:
        areas = os.listdir(os.path.join(grid_search_results_dir, session, norm_folder))

    session_results = {}
    for area in areas:
        df = pd.DataFrame({'window': [], 'matrix_size': [], 'r': [], 'AICs': [], 'time_vals': [], 'file_paths': []}).set_index(['window', 'matrix_size', 'r'])
        area_folder = os.path.join(grid_search_results_dir, session, norm_folder, area)
        for f in os.listdir(area_folder):
            t = float(f.split('_')[0])
            file_path = os.path.join(area_folder, f)
            df_new = pd.DataFrame(pd.read_pickle(file_path))
            if np.isnan(df_new.AIC).sum() > 0:
                print(file_path)
            df_new = df_new.set_index(['window', 'matrix_size', 'r'])
            for i, row in df_new.iterrows():
                if i in df.index:
                    df.loc[i, 'AICs'].append(row.AIC)
                    df.loc[i, 'time_vals'].append(t)
                    df.loc[i, 'file_paths'].append(file_path)
                else:
                    df.loc[i] = {'AICs': [row.AIC], 'time_vals': [t], 'file_paths': [file_path]}

        df = df.loc[df.index.sortlevel()[0]]
        session_results[area] = df
    
    return session_results

def combine_grid_results(results_dict):
    all_results = None
    for key, results in results_dict.items():
        if all_results is None:
            all_results = deepcopy(results)
            if 'AICs' not in all_results.columns:
                all_results['AICs'] = all_results.AIC.apply(lambda x: [x])
                all_results = all_results.drop('AIC', axis='columns')
        else:
            for i, row in results.iterrows():
                if i in all_results.index:
                    if 'AICs' in row:
                        all_results.loc[i, 'AICs'].extend(row.AICs)
                    else:
                        all_results.loc[i, 'AICs'].append(row.AIC)
                    if 'time_vals' in all_results.columns:
                        all_results.loc[i, 'time_vals'].extend(row.time_vals)
                    if 'file_paths' in all_results.columns:
                        all_results.loc[i, 'file_paths'].extend(row.file_paths)
                else:
                    if 'AICs' in row:
                        all_results.loc[i] = {'AICs': row.AICs, 'time_vals': row.time_vals, 'file_paths': row.file_paths}
                    else:
                        all_results.loc[i] = {'AICs': [row.AIC], 'time_vals': row.time_vals, 'file_paths': row.file_paths}
#     full_length_inds = all_results.AICs.apply(lambda x: len(x)) == all_results.AICs.apply(lambda x: len(x)).max()
#     window, matrix_size, r = all_results.index[full_length_inds][all_results[full_length_inds].AICs.apply(lambda x: np.mean(x)).argmin()]
    
#     all_results = all_results.drop(all_results[all_results.index.get_level_values('matrix_size') < all_results.index.get_level_values('r')].index, inplace=False)
#     window, matrix_size, r = all_results.index[all_results.AICs.apply(lambda x: np.mean(x)).argmin()]
    
    while True:
        opt_index = all_results.index[all_results.AICs.apply(lambda x: np.mean(x)).argmin()]
        in_all_dfs = True
        for key, result in results_dict.items():
            if opt_index not in result.index:
                in_all_dfs = False
                break

        if in_all_dfs:
            break
        else:
            all_results = all_results.drop(opt_index, inplace=False)
    
    window, matrix_size, r = opt_index

    return window, matrix_size, r, all_results

# def get_chosen_params(session, results_dir, grid_search_results_dir, normed=False):
#     chosen_params_dir = os.path.join(results_dir, 'chosen_params')
#     os.makedirs(chosen_params_dir, exist_ok=True)
#     chosen_params_filepath = os.path.join(chosen_params_dir, session)
#     if os.path.exists(chosen_params_filepath):
#         chosen_params = pd.read_pickle(chosen_params_filepath)
#     else:
#         session_grid_results = compile_grid_results(session, grid_search_results_dir, normed=normed)
#         chosen_params = {}
#         for area in session_grid_results.keys():
#             window, matrix_size, r, all_results = combine_grid_results({area: session_grid_results[area]})
#             chosen_params[area] = dict(
#                 window=window,
#                 matrix_size=matrix_size,
#                 r=r
#             )
#         pd.to_pickle(chosen_params, chosen_params_filepath)
    
#     return chosen_params

def get_section_info(session, all_data_dir, data_class):
    if 'propofol' in data_class:
        section_info = [('awake', [-15, 0]), ('induction', [0, 15]), ('loading dose', [15, 30]), ('maintenance dose', [30, 60]), ('recovery', [60, 75])]
        section_info_extended = [('awake', [-15, 0]), ('induction', [0, 15]), ('unconscious', [15, 60]), ('late recovery', [68, 75]), ('recovery', [60, 75]), ('loading dose', [15, 30]), ('maintenance dose', [45, 60]), ('early recovery', [60, 68])]

        section_colors = {
            'awake': 'limegreen',
            'unconscious': 'plum',
            'recovery': 'orange',
            'loading dose': '#D65CD4',
            'maintenance dose': '#E28DE0',
            'early recovery': 'chocolate',
            'late recovery': '#FFBF47',
            'induction': '#61C9A8'
        }
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        session_file = h5py.File(os.path.join(all_data_dir, 'anesthesia', 'mat', data_class, session + '.mat'), 'r')
        infusion_start = session_file['sessionInfo']['drugStart'][0, 0]

    elif 'lever' in data_class or 'Lvr' in data_class:
        # os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        # variables = ['sessionInfo', 'trialInfo']
        # session_vars, T, N, dt = load_session_data(session, all_data_dir, variables, data_class=data_class, verbose=False)
        # session_info, trial_info = session_vars['sessionInfo'], session_vars['trialInfo']

        session_file = h5py.File(os.path.join(all_data_dir, data_class, 'mat', session + '.mat'), 'r')
        infusion_start = session_file['sessionInfo']['infusionStart'][0, 0]
        tasks = convert_h5_string_array(session_file, session_file['trialInfo']['task'])
        trial_starts = session_file['trialInfo']['trialStart'][:, 0]

        # get odbball windows
        oddball_windows = []
        oddball_on = False

        for ind in range(len(tasks)):

            t = trial_starts[ind]
            task = tasks[ind]
            
            if task == 'oddball' and not oddball_on:
                oddball_on = True
                oddball_windows.append([(t - infusion_start)/60, np.inf])
            if task == 'lever' and oddball_on:
                oddball_on = False
                oddball_windows[-1][-1] = (t - infusion_start)/60
            if ind == len(tasks) - 1:
                oddball_windows[-1][-1] = (t - infusion_start)/60

        # section_info = [
        #                 ('awake', [-36, 0]), 
        #                 ('induction', [0, oddball_windows[1][0]]), 
        #                 ('anesthesia oddball', oddball_windows[1]), 
        #                 ('anesthesia', [oddball_windows[1][1], (oddball_windows[2][0] - oddball_windows[1][1])/2]), 
        #                 ('late anesthesia', [(oddball_windows[2][0] - oddball_windows[1][1])/2, oddball_windows[2][1]]), 
        #                 ('recovery oddball', oddball_windows[2])
        #             ]
        section_info = [
                        ('awake lever1', [-infusion_start/60, oddball_windows[0][0]]),
                        ('awake oddball', oddball_windows[0]),
                        ('awake lever2', [oddball_windows[0][1], 0]), 
                        ('induction', [0, oddball_windows[1][0]]), 
                        ('unconscious oddball', oddball_windows[1]), 
                        ('early unconscious', [oddball_windows[1][1], 45]), 
                        # ('early unconscious', [oddball_windows[1][1], (oddball_windows[2][0] - oddball_windows[1][1])/2]), 
                        # ('late unconscious', [(oddball_windows[2][0] - oddball_windows[1][1])/2, oddball_windows[2][1]]), 
                        ('late unconscious', [45, oddball_windows[2][1]]),
                        ('recovery oddball', oddball_windows[2])
                    ]
        section_info_extended = section_info
        section_colors = {
            'awake lever1': 'limegreen',
            'awake lever2': 'limegreen',
            'awake oddball': 'forestgreen',
            'unconscious oddball': 'plum',
            'recovery oddball': 'orange',
            'early unconscious': '#D65CD4',
            'late unconscious': '#E28DE0',
            'induction': '#61C9A8'
        }
        
    return section_info, section_info_extended, section_colors, infusion_start


def get_grid_search_window_ts(cfg, session, valid_window_starts=None, window_radius=None, random_state=None):


    section_info, section_info_extended, section_colors, infusion_start = get_section_info(session, cfg.params.all_data_dir, cfg.params.data_class)
    num_windows_per_section = cfg.params.num_windows_per_section
    window_radius = 2*cfg.params.window if window_radius is None else window_radius
    if 'propofol' in cfg.params.data_class:
        session_file = h5py.File(os.path.join(cfg.params.all_data_dir, 'anesthesia', 'mat', cfg.params.data_class, session + '.mat'), 'r')
    else:
        session_file = h5py.File(os.path.join(cfg.params.all_data_dir, cfg.params.data_class, 'mat', session + '.mat'), 'r')
    drug_start = get_drug_start(cfg, session_file)
    
    if random_state is not None:
        np.random.seed(random_state)
    grid_search_window_start_ts = np.zeros(num_windows_per_section*len(section_info))

    if valid_window_starts is None:
        ind = 0
        for section, times in section_info:
            for i in range(num_windows_per_section):
                t = np.random.uniform(times[0], times[1] - window_radius/60)
                t = t*60 + drug_start
                if ind > 0:
                    while np.abs(grid_search_window_start_ts[:ind] - t).min() <= window_radius: # if the chosen time is within window_radius seconds of already chosen times
                        t = np.random.uniform(times[0], times[1] - window_radius/60)
                        t = t*60 + drug_start
                grid_search_window_start_ts[ind] = t
                ind += 1
    else:
        valid_window_starts = np.array(valid_window_starts)
        ind = 0
        for section, times in section_info:
            section_inds = (((valid_window_starts - drug_start)/60) >= times[0]) & (((valid_window_starts - drug_start)/60) <= times[1])
            if np.sum(section_inds) < num_windows_per_section:
                print(f"Only {np.sum(section_inds)} valid windows could be found for section '{section}' with times {times}")
                grid_search_window_start_ts = grid_search_window_start_ts[:-(num_windows_per_section - np.sum(section_inds))]
            num_windows_to_choose = np.min([num_windows_per_section, np.sum(section_inds)])
            grid_search_window_start_ts[ind:ind + num_windows_to_choose] = np.random.choice(valid_window_starts[section_inds], size=num_windows_to_choose, replace=False)
            ind += num_windows_to_choose
    grid_search_window_start_ts = np.sort(grid_search_window_start_ts)

    return grid_search_window_start_ts

def get_grid_search_run_list(cfg, session, grid_search_window_start_ts=None, bad_electrodes=[], verbose=False, make_new=False):
    grid_search_run_list_dir = os.path.join(cfg.params.grid_search_results_dir, cfg.params.data_class, 'grid_search_run_lists')
    os.makedirs(grid_search_run_list_dir, exist_ok=True)
    grid_search_run_list_file = os.path.join(grid_search_run_list_dir, f"{session}_{cfg.params.grid_set}_window_{cfg.params.window}")
    
    if os.path.exists(grid_search_run_list_file) and not make_new:
        if verbose:
            print(f"list exists! loading {grid_search_run_list_file}...")
        grid_search_run_list = pd.read_pickle(grid_search_run_list_file)

    # MAKE THE LIST
    else:
        if grid_search_window_start_ts is None:
            raise ValueError(f"File {grid_search_run_list_file} was not found and grid_search_window_start_ts is needed to generate a new list")

        # os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        # variables = ['electrodeInfo', 'lfpSchema', 'sessionInfo']
        # session_vars, T, N, dt = load_session_data(session, all_data_dir, variables, data_class=data_class, verbose=False)
        # electrode_info, lfp_schema, session_info = session_vars['electrodeInfo'], session_vars['lfpSchema'], session_vars['sessionInfo']

        if 'propofol' in cfg.params.data_class:
            session_file = h5py.File(os.path.join(cfg.params.all_data_dir, 'anesthesia', 'mat', cfg.params.data_class, session + '.mat'), 'r')
        else:
            session_file = h5py.File(os.path.join(cfg.params.all_data_dir, cfg.params.data_class, 'mat', session + '.mat'), 'r')
        electrode_areas = convert_h5_string_array(session_file, session_file['electrodeInfo']['area'])
        if len(electrode_areas) > session_file['lfp'].shape[0]:
            electrode_areas = np.delete(electrode_areas, 60)
        
        areas = np.unique(electrode_areas)
        areas = np.concatenate((areas, ('all',)))
        if np.sum(['-L' in area for area in areas]) > 0:
            areas_bilateral = np.unique([area.split('-')[0] for area in areas if area!='all' and ('-L' in area or '-R' in area)])
            areas = np.concatenate((areas, areas_bilateral))
            areas = np.concatenate((areas, ['-L', '-R']))
        dt = session_file['lfpSchema']['smpInterval'][0, 0]

        grid_search_run_list = {}

        for area in areas:
            grid_search_run_list[area] = []
            
            if cfg.params.T_pred is None:
                T_pred = cfg.params.window
            else:
                T_pred = cfg.params.T_pred
        
            if area == 'all':
                unit_indices = np.arange(len(electrode_areas))
            else:
                # unit_indices = np.where(electrode_info['area'] == area)[0]
                unit_indices = np.where([area in area_entry for area_entry in electrode_areas])[0]
            unit_indices = np.array([idx for idx in unit_indices if idx not in bad_electrodes])

            for window_num, window_start in enumerate(grid_search_window_start_ts):
                for i, n_delays in enumerate(cfg.grid_sets[cfg.params.grid_set].n_delays_vals):
                    if cfg.params.group_ranks:
                        grid_search_run_list[area].append(dict(
                            session=session,
                            area=area,
                            window_start=window_start,
                            window_end=window_start + cfg.params.window,
                            test_window_start=window_start + cfg.params.window,
                            test_window_end=window_start + cfg.params.window + T_pred,
                            n_delays=n_delays,
                            rank=list(cfg.grid_sets[cfg.params.grid_set].rank_vals),
                            dimension_inds=unit_indices,
                            window_num=window_num,
                            i=i,
                            dt=dt,
                        ))
                    else:
                        for j, rank in enumerate(cfg.grid_sets[cfg.params.grid_set].rank_vals):
                            grid_search_run_list[area].append(dict(
                                session=session,
                                area=area,
                                window_start=window_start,
                                window_end=window_start + cfg.params.window,
                                test_window_start=window_start + cfg.params.window,
                                test_window_end=window_start + cfg.params.window + T_pred,
                                n_delays=n_delays,
                                rank=rank,
                                dimension_inds=unit_indices,
                                window_num=window_num,
                                i=i,
                                j=j,
                                dt=dt,
                            ))
    
        pd.to_pickle(grid_search_run_list, grid_search_run_list_file)
    
    return grid_search_run_list

def get_grid_search_results(cfg, session_list, areas, num_sections, pca_chosen=None, metric_keys=['mase', 'r2', 'aic', 'mse'], verbose=False):
    grid_search_results = {}
    for session in session_list:
        if verbose:
            print("-"*20)
            print(f"SESSION = {session}")
            print("-"*20)
        noise_filter_folder = f"NOISE_FILTERED_{cfg.params.window}_{cfg.params.wake_amplitude_thresh}_{cfg.params.anesthesia_amplitude_thresh}_{cfg.params.electrode_num_thresh}" if cfg.params.noise_filter else "NO_NOISE_FILTER"
        normed_folder = 'NOT_NORMED' if not cfg.params.normed else 'NORMED'
        filter_folder = f"[{cfg.params.high_pass},{cfg.params.low_pass}]" if cfg.params.low_pass is not None or cfg.params.high_pass is not None else 'NO_FILTER'

        grid_search_results[session] = {}
        for area in areas:
            grid_search_results[session][area] = {}
            grid_search_results[session][area]['mats'] = {}
            for key in metric_keys:
                grid_search_results[session][area]['mats'][key] = np.zeros((cfg.params.num_windows_per_section*num_sections, len(cfg.grid_sets[cfg.params.grid_set].n_delays_vals), len(cfg.grid_sets[cfg.params.grid_set].rank_vals)))
            
            grid_search_run_list = get_grid_search_run_list(cfg, session, verbose=verbose)
            pca_folder = "NO_PCA" if not cfg.params.pca else f"PCA_{pca_chosen[session][area]}"
            save_dir = os.path.join(cfg.params.grid_search_results_dir, cfg.params.data_class, 'grid_search_results', session, noise_filter_folder, normed_folder, f"SUBSAMPLE_{cfg.params.subsample}", filter_folder, f"WINDOW_{cfg.params.window}", cfg.params.grid_set, area, pca_folder)
            
            if verbose:
                print(f"Loading data for {session} - {area}")
            window_starts = set()
            for f in tqdm(os.listdir(save_dir), disable=not verbose):
                ret = pd.read_pickle(os.path.join(save_dir, f))
                run_index = int(f.split('-')[1].split('.')[0])
                run_info = grid_search_run_list[area][run_index]
     
                for key in metric_keys:
                    if cfg.params.group_ranks:
                        grid_search_results[session][area]['mats'][key][run_info['window_num'], run_info['i']] = [ret_[key] for ret_ in ret]
                    else:
                        grid_search_results[session][area]['mats'][key][run_info['window_num'], run_info['i'], run_info['j']] = ret[key]

                window_starts.add(run_info['window_start'])
                # window_starts.append(run_info['window_start'])

            i, j = np.unravel_index(np.nanargmin(grid_search_results[session][area]['mats'][cfg.params.stat_to_use].mean(axis=0)), shape=grid_search_results[session][area]['mats'][cfg.params.stat_to_use].shape[1:])
            n_delays = cfg.grid_sets[cfg.params.grid_set].n_delays_vals[int(i)]
            rank = cfg.grid_sets[cfg.params.grid_set].rank_vals[int(j)]

            grid_search_results[session][area]['i'] = i
            grid_search_results[session][area]['j'] = j
            grid_search_results[session][area]['n_delays'] = n_delays
            grid_search_results[session][area]['rank'] = rank
            grid_search_results[session][area]['window_start_ts'] = window_starts

    return grid_search_results

def get_delase_run_list(cfg, session, valid_window_starts=None, bad_electrodes=[], verbose=False):
    delase_run_list_dir =  os.path.join(cfg.params.delase_results_dir, cfg.params.data_class, 'delase_run_lists')
    os.makedirs(delase_run_list_dir, exist_ok=True)
    sections_to_use_folder = 'SECTIONS_TO_USE_' + '__'.join(['_'.join(section.split(' ')) for section in cfg.params.sections_to_use])
    delase_run_list_file = os.path.join(delase_run_list_dir, f"{session}__stride_{cfg.params.stride}__window_{cfg.params.window}__{sections_to_use_folder}") 

    if os.path.exists(delase_run_list_file):
        delase_run_list = pd.read_pickle(delase_run_list_file)

    # MAKE THE LIST
    else:

        # GET SECTION AND SESSION INFO
        section_info, section_info_extended, section_colors, infusion_start = get_section_info(session, cfg.params.all_data_dir, cfg.params.data_class)

        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        variables = ['electrodeInfo', 'lfpSchema', 'sessionInfo']
        session_vars, T, N, dt = load_session_data(session, cfg.params.all_data_dir, variables, data_class=cfg.params.data_class, verbose=False)
        electrode_info, lfp_schema, session_info = session_vars['electrodeInfo'], session_vars['lfpSchema'], session_vars['sessionInfo']
        areas = np.unique(electrode_info['area'])
        areas = np.concatenate((areas, ('all',)))
        if np.sum(['-L' in area for area in areas]) > 0:
            areas_bilateral = np.unique([area.split('-')[0] for area in areas if area!='all' and ('-L' in area or '-R' in area)])
            areas = np.concatenate((areas, areas_bilateral))
            areas = np.concatenate((areas, ['-L', '-R']))

        if 'propofol' in cfg.params.data_class:
            drug_start = session_info['drugStart'][0]
        elif 'lever' in cfg.params.data_class or 'Lvr' in cfg.params.data_class:
            drug_start = session_info['infusionStart']

        directory_path = os.path.join(cfg.params.all_data_dir, cfg.params.data_class, session + '_lfp_chunked_20s', 'directory')

        delase_run_list = {}

        for area in areas:
            delase_run_list[area] = []

            window = cfg.params.window
            if cfg.params.stride is None:
                stride = window
            else:
                stride = cfg.params.stride
            if cfg.params.T_pred is None:
                T_pred = window
            else:
                T_pred = cfg.params.T_pred
        
            if area == 'all':
                unit_indices = np.arange(len(electrode_info['area']))
            else:
                # unit_indices = np.where(electrode_info['area'] == area)[0]
                unit_indices = np.where([area in area_entry for area_entry in electrode_info['area']])[0]
            unit_indices = np.array([idx for idx in unit_indices if idx not in bad_electrodes])

            if 'all_sections' in cfg.params.sections_to_use:
                if valid_window_starts is None:
                    num_windows = int(np.floor((T - ((window + T_pred)/dt))/(stride/dt)) + 1)
                    window_start_times = np.arange(num_windows)*stride

                    # if min_time is None:
                    #     min_time = -drug_start/60
                    # if max_time is None:
                    #     max_time = T*dt/60 - drug_start/60
                    # window_start_times = window_start_times[((window_start_times - drug_start)/60 >= min_time) & ((window_start_times - drug_start)/60 <= max_time)]
                else:
                    window_start_times = np.array(valid_window_starts)

            else:
                if valid_window_starts is None:
                    window_start_times = np.array([])
                    for section_key, section_times in section_info:
                        if section_key in cfg.params.sections_to_use:
                            section_start = section_times[0]*60 + drug_start
                            section_end = section_times[1]*60 + drug_start
                            num_windows = int(np.floor((section_end - section_start - (window + T_pred))/(stride)) + 1)
                            window_start_times = np.concatenate((window_start_times, np.arange(num_windows)*stride + section_start))
                else:
                    valid_window_starts = np.array(valid_window_starts)
                    for section_key, section_times in section_info:
                        if section_key in cfg.params.sections_to_use:
                            section_start = section_times[0]*60 + drug_start
                            section_end = section_times[1]*60 + drug_start
                            section_inds = (valid_window_starts >= section_start) & (valid_window_starts <= section_end)
                            window_start_times = np.concatenate((window_start_times, valid_window_starts[section_inds]))

            iterator = tqdm(total = int(len(window_start_times)), desc=f"Creating DeLASE run list for {area}", disable=not verbose)

            for i, window_start in enumerate(window_start_times):
                delase_run_list[area].append(dict(
                    session=session,
                    area=area,
                    window_start=window_start,
                    window_end=window_start + window,
                    test_window_start=window_start + window,
                    test_window_end=window_start + window + T_pred,
                    dimension_inds=unit_indices,
                    directory_path=directory_path,
                    stride=stride,
                    i=i,
                    dt=dt,
                    n=len(window_start_times)
                ))

                iterator.update()
            iterator.close()

        pd.to_pickle(delase_run_list, delase_run_list_file)
    
    return delase_run_list

def get_delase_results(cfg, session_list, areas, grid_params_to_use, pca_chosen=None, verbose=False):
    delase_results = {}
    for session in session_list:
        if verbose:
            print("-"*20)
            print(f"SESSION = {session}")
            print("-"*20)
        noise_filter_folder = f"NOISE_FILTERED_{cfg.params.window}_{cfg.params.wake_amplitude_thresh}_{cfg.params.anesthesia_amplitude_thresh}_{cfg.params.electrode_num_thresh}" if cfg.params.noise_filter else "NO_NOISE_FILTER"
        normed_folder = 'NOT_NORMED' if not cfg.params.normed else 'NORMED'
        filter_folder = f"[{cfg.params.high_pass},{cfg.params.low_pass}]" if cfg.params.low_pass is not None or cfg.params.high_pass is not None else 'NO_FILTER'

        normed_folder = 'NOT_NORMED' if not cfg.params.normed else 'NORMED'
        filter_folder = f"[{cfg.params.high_pass},{cfg.params.low_pass}]" if cfg.params.low_pass is not None or cfg.params.high_pass is not None else 'NO_FILTER'
        sections_to_use_folder = 'SECTIONS_TO_USE_' + '__'.join(['_'.join(section.split(' ')) for section in cfg.params.sections_to_use])
        # delase_run_list = get_delase_run_list(cfg, session)

        delase_results[session] = {}
        for area in areas:
            delase_results[session][area] = []
            grid_folder = f"GRID_RESULTS_n_delays_{grid_params_to_use[session][area]['n_delays']}_rank_{grid_params_to_use[session][area]['rank']}"
            pca_folder = "NO_PCA" if not cfg.params.pca else f"PCA_{pca_chosen[session][area]}"
            save_dir = os.path.join(cfg.params.delase_results_dir, cfg.params.data_class, 'delase_results', session, noise_filter_folder, normed_folder, f"SUBSAMPLE_{cfg.params.subsample}", filter_folder, f"WINDOW_{cfg.params.window}", grid_folder, f"STRIDE_{cfg.params.stride}", area, pca_folder)
            
            if verbose:
                print(f"Loading data for {session} - {area}")
            for f in tqdm(os.listdir(save_dir), disable=not verbose):
                ret = pd.read_pickle(os.path.join(save_dir, f))
                run_index = int(f.split('-')[1].split('.')[0])
                # run_info = delase_run_list[area][run_index]

                ret['run_index'] = run_index

                delase_results[session][area].append(ret)
            if len(delase_results[session][area]) == 0:
                print(f"No results found for {session} - {area}, grid params: {grid_params_to_use[session][area]}")
            delase_results[session][area] = pd.DataFrame(delase_results[session][area]).sort_values('window_start').reset_index()

    return delase_results

def resection_grid_results(cfg, grid_search_results, sections_to_use=['all_sections']):
    if 'all_sections' in sections_to_use:
        return grid_search_results

    for session in grid_search_results.keys():
        session_vars, T, N, dt = load_session_data(session, cfg.params.all_data_dir, ['sessionInfo'], data_class=cfg.params.data_class, verbose=False)
        session_info = session_vars['sessionInfo']

        section_info, section_info_extended, section_colors = get_section_info(session, cfg.params.all_data_dir, cfg.params.data_class)

        for area in grid_search_results[session].keys():

            window_start_ts = grid_search_results[session][area]['window_start_ts']
            window_inds_to_use = []
            window_starts = []
            for ind, t in enumerate(window_start_ts):
                for section_key, section_times in section_info:
                    # print(section_key)
                    if section_key in sections_to_use:
                        if t/60 - session_info['infusionStart']/60 >= section_times[0] and t/60 - session_info['infusionStart']/60 < section_times[1]:
                            window_inds_to_use.append(ind)
                            window_starts.append(t)
            
            for stat in grid_search_results[session][area]['mats'].keys():
                grid_search_results[session][area]['mats'][stat] = grid_search_results[session][area]['mats'][stat][window_inds_to_use, :, :]

            i, j = np.unravel_index(np.nanargmin(grid_search_results[session][area]['mats'][cfg.params.stat_to_use].mean(axis=0)), shape=grid_search_results[session][area]['mats'][cfg.params.stat_to_use].shape[1:])
            n_delays = cfg.grid_sets[cfg.params.grid_set].n_delays_vals[int(i)]
            rank = cfg.grid_sets[cfg.params.grid_set].rank_vals[int(j)]

            grid_search_results[session][area]['i'] = i
            grid_search_results[session][area]['j'] = j
            grid_search_results[session][area]['n_delays'] = n_delays
            grid_search_results[session][area]['rank'] = rank
            grid_search_results[session][area]['window_start_ts'] = window_starts
    
    return grid_search_results

def butter_highpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    # return b, a
    sos = signal.butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    return sos

def butter_highpass_filter(data, cutoff, fs, order=2, bidirectional=True):
    # b, a = butter_highpass(cutoff, fs, order=order)
    # y = signal.filtfilt(b, a, data)
    sos = butter_highpass(cutoff, fs, order=order)
    if bidirectional:
        y = signal.sosfiltfilt(sos, data)
    else:
        y = signal.sosfilt(sos, data)
    return y

def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    # return b, a
    sos = signal.butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    return sos

def butter_lowpass_filter(data, cutoff, fs, order=2, bidirectional=True):
    # b, a = butter_lowpass(cutoff, fs, order=order)
    # y = signal.filtfilt(b, a, data)
    sos = butter_lowpass(cutoff, fs, order=order)
    if bidirectional:
        y = signal.sosfiltfilt(sos, data)
    else:
        y = signal.sosfilt(sos, data)
    return y

# Define the bandstop filter function
def butter_bandstop_filter(data, lowcut, highcut, fs, order=2, bidirectional=True):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    # b, a = signal.butter(order, [low, high], btype='bandstop')
    # y = signal.filtfilt(b, a, data)
    sos = signal.butter(order, [low, high], btype='bandstop', output='sos')
    if bidirectional:
        y = signal.sosfiltfilt(sos, data)
    else:
        y = signal.sosfilt(sos, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # b, a = signal.butter(order, [low, high], btype='band')
    # return b, a
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2, bidirectional=True):
    # b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # y = signal.lfilter(b, a, data)
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    if bidirectional:
        y = signal.sosfiltfilt(sos, data)
    else:
        y = signal.sosfilt(sos, data)
    return y

def filter_data(data, low_pass=None, high_pass=None, dt=0.001, order=2, bidirectional=True):
    if low_pass is None and high_pass is None:
        return data
    elif low_pass is None and high_pass is not None:
        data_filt = np.zeros(data.shape)
        for i in range(data.shape[1]):
            data_filt[:, i] = butter_highpass_filter(data[:, i], high_pass, 1/dt, order=order, bidirectional=bidirectional)
        return data_filt
    elif low_pass is not None and high_pass is None:
        data_filt = np.zeros(data.shape)
        for i in range(data.shape[1]):
            data_filt[:, i] = butter_lowpass_filter(data[:, i], low_pass, 1/dt, order=order, bidirectional=bidirectional)
        return data_filt
    else:
        if low_pass == high_pass:
            return data
        elif low_pass > high_pass:
            data_filt = np.zeros(data.shape)
            for i in range(data.shape[1]):
                data_filt[:, i] = butter_bandpass_filter(data[:, i], high_pass, low_pass, 1/dt, order=order, bidirectional=bidirectional)
            return data_filt
        else: # low_pass < high_pass
            data_filt = np.zeros(data.shape)
            for i in range(data.shape[1]):
                data_filt[:, i] = butter_bandstop_filter(data[:, i], low_pass, high_pass, 1/dt, order=order, bidirectional=bidirectional)
            return data_filt

def get_drug_start(cfg, session_file):
    return session_file['sessionInfo']['drugStart'][0] if 'propofol' in cfg.params.data_class else session_file['sessionInfo']['infusionStart'][0, 0]

def get_loc_roc(cfg, session_file, read_file=True):
    session = convert_char_array(session_file['sessionInfo']['session'])
    save_file = os.path.join(cfg.params.results_dir, 'loc_roc_info', cfg.params.data_class, f"{session}.pkl")
    if read_file and os.path.exists(save_file):
        ret_dict = pd.read_pickle(save_file)
        return ret_dict['loc'], ret_dict['roc'], ret_dict['ropap']

    if 'propofol' in cfg.params.data_class:
        loc = session_file['sessionInfo']['eyesClose'][0, -1] 
        roc = session_file['sessionInfo']['eyesOpen'][0, -1]
        ropap = roc

        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        pd.to_pickle(dict(loc=loc, roc=roc, ropap=ropap), save_file)
    elif 'lever' in cfg.params.data_class or 'Lvr' in cfg.params.data_class:
        lever_window = 60
        stride = 0.1
        loc_thresh = 0
        roc_thresh = 0
        ropap_thresh = 0.25

        trial_starts = session_file['trialInfo']['trialStart'][:,0]
        trial_ends = session_file['trialInfo']['trialEnd'][:,0]
        tasks = convert_h5_string_array(session_file, session_file['trialInfo']['task'])

        lever_starts = trial_starts[tasks == 'lever']
        lever_ends = trial_ends[tasks == 'lever']
        window_starts = np.arange(0, lever_ends[-1], stride)

        drug_start = get_drug_start(cfg, session_file)
        pct_correct = np.zeros(len(window_starts))
        loc = int(window_starts[(window_starts >= drug_start) & (pct_correct <= loc_thresh)].min())

        for i, window_start in enumerate(window_starts):
            window_end = window_start + lever_window
            outcomes = session_file['trialInfo']['lvr_correct'][()][(lever_starts >= window_start) & (lever_ends <= window_end)]
            if len(outcomes) > 0:
                pct_correct[i] = outcomes.sum()/len(outcomes)

        roc = int(window_starts[(window_starts > loc + 10*60) & (pct_correct > roc_thresh)].min())
        valid_windows = window_starts[(window_starts > loc + 10*60) & (pct_correct > ropap_thresh)]
        ropap = int(valid_windows.min() if len(valid_windows) > 0 else window_starts[-1])

        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        pd.to_pickle(dict(loc=loc, roc=roc, ropap=ropap), save_file)

        # plt.plot((window_starts - drug_start)/60, pct_correct)
        # plt.axvline(0, c='plum',label='Drug Start', linestyle='--')
        # plt.axvline((loc - drug_start)/60, c='r', label='LOC', linestyle='--')
        # plt.axvline((roc - drug_start)/60, c='g', label='ROC', linestyle='--')
        # # plt.axvline((ropap - drug_start)/60, c='orange', label='ROPAP', linestyle='--')
        # plt.xlabel('Time Relative to Anesthesia Start (min)')
        # plt.ylabel('Response Probability')
        # plt.legend()
        # plt.show()
    
    return loc, roc, ropap

def get_pct_correct(cfg, session_file, lever_window=60, stride=0.1):
    if 'propofol' in cfg.params.data_class:
        raise NotImplementedError("Propofol doesn't have a pct_correct")
    elif 'lever' in cfg.params.data_class or 'Lvr' in cfg.params.data_class:
        trial_starts = session_file['trialInfo']['trialStart'][:,0]

        trial_starts = session_file['trialInfo']['trialStart'][:,0]
        trial_ends = session_file['trialInfo']['trialEnd'][:,0]
        tasks = convert_h5_string_array(session_file, session_file['trialInfo']['task'])

        lever_starts = trial_starts[tasks == 'lever']
        lever_ends = trial_ends[tasks == 'lever']
        window_starts = np.arange(0, lever_ends[-1], stride)

        drug_start = get_drug_start(cfg, session_file)
        pct_correct = np.zeros(len(window_starts))

        for i, window_start in enumerate(window_starts):
            window_end = window_start + lever_window
            outcomes = session_file['trialInfo']['lvr_correct'][()][(lever_starts >= window_start) & (lever_ends <= window_end)]
            if len(outcomes) > 0:
                pct_correct[i] = outcomes.sum()/len(outcomes)

        # roc = int(window_starts[(window_starts > loc + 10*60) & (pct_correct > roc_thresh)].min())
        # valid_windows = window_starts[(window_starts > loc + 10*60) & (pct_correct > ropap_thresh)]
        # ropap = int(valid_windows.min() if len(valid_windows) > 0 else window_starts[-1])

        # plt.plot((window_starts - drug_start)/60, pct_correct)
        # plt.axvline(0, c='plum',label='Drug Start', linestyle='--')
        # plt.axvline((loc - drug_start)/60, c='r', label='LOC', linestyle='--')
        # plt.axvline((roc - drug_start)/60, c='g', label='ROC', linestyle='--')
        # # plt.axvline((ropap - drug_start)/60, c='orange', label='ROPAP', linestyle='--')
        # plt.xlabel('Time Relative to Anesthesia Start (min)')
        # plt.ylabel('Response Probability')
        # plt.legend()
        # plt.show()
    
    return pct_correct, window_starts

def find_noisy_data(cfg, session, return_all=False):

    if 'propofol' in cfg.params.data_class:
        filename = os.path.join(cfg.params.all_data_dir, 'anesthesia', 'mat', cfg.params.data_class, f"{session}.mat")
    else:
        filename = os.path.join(cfg.params.all_data_dir, cfg.params.data_class, 'mat', f"{session}.mat")
    session_file = h5py.File(filename, 'r')

    loc, roc, ropap = get_loc_roc(cfg, session_file)

    window = cfg.params.window
    wake_amplitude_thresh = cfg.params.wake_amplitude_thresh
    anesthesia_amplitude_thresh = cfg.params.anesthesia_amplitude_thresh
    electrode_num_thresh = cfg.params.electrode_num_thresh

    T = session_file['lfp'].shape[1]
    dt = session_file['lfpSchema']['smpInterval'][0, 0]

    window_starts = np.arange(0, T*dt - window, window)
    window_df = []

    lfp = TransposedDatasetView(session_file['lfp']).transpose()

    for i, window_start in tqdm(enumerate(window_starts), total=len(window_starts)):
        window_end = window_start + window
        window_data = lfp[int(window_start/dt):int((window_end)/dt)]
        max_vals = (np.abs(window_data)).max(axis=0)
        if window_start <= loc or window_start >= roc:
            bad_electrodes = np.where(max_vals > wake_amplitude_thresh)[0]
            wake = True
        else:
            bad_electrodes = np.where(max_vals > anesthesia_amplitude_thresh)[0]
            wake = False 
        window_df.append(dict(
            window_start=window_start,
            window_end=window_end,
            max_vals=max_vals,
            bad_electrodes=bad_electrodes,
            num_bad_electrodes=len(bad_electrodes),
            noise_window = len(bad_electrodes) > electrode_num_thresh,
            wake=wake
        ))
    window_df = pd.DataFrame(window_df)

    bad_electrodes = np.sort(np.unique(np.hstack(window_df[~window_df.noise_window].bad_electrodes.to_numpy())))
    # ensure that any window that is immediately followed by a noisy window is not included
    def find_false_true_pairs(bool_vector):
        # Use zip to pair each element with its next element
        return np.concat((np.array([current is False and next is True for i, (current, next) in enumerate(zip(bool_vector[:-1], bool_vector[1:]))]), [True]))
    noise_precedes = find_false_true_pairs(window_df.noise_window)
    valid_window_inds = (~noise_precedes) & (~window_df.noise_window)
    valid_window_starts = window_df[valid_window_inds].window_start.to_numpy()

    if return_all:
        drug_start = get_drug_start(cfg, session_file)
        plot_kwargs = dict(
            lfp=lfp,
            dt=dt,
            drug_start=drug_start,
            loc=loc,
            roc=roc,
            window=window,
            wake_amplitude_thresh=wake_amplitude_thresh,
            anesthesia_amplitude_thresh=anesthesia_amplitude_thresh,
            electrode_num_thresh=electrode_num_thresh,
            session=session,
        )
        return window_df, bad_electrodes, valid_window_starts, plot_kwargs
    return window_df, bad_electrodes, valid_window_starts

# --------------------------------------------------------------------------
# Multiprocessing code
# --------------------------------------------------------------------------

def log_msg(msg, log=None, verbose=False):
    if log is not None:
        log.info(msg)
    elif verbose:
        print(msg)

def get_noise_filter_info(cfg, session_list, log=None, verbose=False):
    noise_filter_info = {}
    if cfg.params.noise_filter:
        os.makedirs(cfg.params.noise_filter_results_dir, exist_ok=True)
        for session in session_list:
            noise_filter_dir = os.path.join(cfg.params.noise_filter_results_dir, cfg.params.data_class)
            os.makedirs(noise_filter_dir, exist_ok=True)
            noise_filter_file = f"{session}__window_{cfg.params.window}__wakethresh_{cfg.params.wake_amplitude_thresh}__anesthesiathresh_{cfg.params.anesthesia_amplitude_thresh}__electrodenum_{cfg.params.electrode_num_thresh}.pkl"
            if noise_filter_file in os.listdir(noise_filter_dir):
                # print(f"File found for session {session}: {os.path.join(noise_filter_dir, noise_filter_file)}")
                noise_filter_info[session] = pd.read_pickle(os.path.join(noise_filter_dir, noise_filter_file))
            else:
                if verbose:
                    log_msg(f"No file for session {session} - finding noisy data...", log, verbose)
                window_df, bad_electrodes, valid_window_starts = find_noisy_data(cfg, session)
                noise_filter_info[session] = dict(
                    window_df=window_df,
                    bad_electrodes=bad_electrodes,
                    valid_window_starts=valid_window_starts
                )
                pd.to_pickle(noise_filter_info[session], os.path.join(noise_filter_dir, noise_filter_file))
            
            if verbose:
                log_msg(f"{len(noise_filter_info[session]['bad_electrodes'])} bad electrodes, {len(noise_filter_info[session]['valid_window_starts'])} valid windows (out of {len(noise_filter_info[session]['window_df'])} total windows)", log, verbose)
    else:
        bad_electrodes = []
        valid_window_starts = None
        for session in session_list:
            noise_filter_info[session] = dict(
                window_df=None,
                bad_electrodes=bad_electrodes,
                valid_window_starts=valid_window_starts
            )
    return noise_filter_info

def get_pca_chosen(cfg, session_list, areas, noise_filter_info, log=None, verbose=False):
    if cfg.params.pca:
        pca_chosen = {}
        for session in session_list:
            section_info, section_info_extended, section_colors, infusion_start = get_section_info(session, cfg.params.all_data_dir, cfg.params.data_class)
            
            normed_folder = 'NOT_NORMED' if not cfg.params.normed else 'NORMED'
            filter_folder = f"[{cfg.params.high_pass},{cfg.params.low_pass}]" if cfg.params.low_pass is not None or cfg.params.high_pass is not None else 'NO_FILTER'
        
            pca_dir = os.path.join(cfg.params.grid_search_results_dir, cfg.params.data_class, "PCA_info", session, normed_folder, f"SUBSAMPLE_{cfg.params.subsample}", filter_folder, f"WINDOW_{cfg.params.window}")
            os.makedirs(pca_dir, exist_ok=True)
        
            pca_chosen[session] = {}
            for area in areas:
                if 'propofol' in cfg.params.data_class:
                    session_file = h5py.File(os.path.join(cfg.params.all_data_dir, 'anesthesia', 'mat', cfg.params.data_class, session + '.mat'), 'r')
                else:
                    session_file = h5py.File(os.path.join(cfg.params.all_data_dir, cfg.params.data_class, 'mat', session + '.mat'), 'r')
                electrode_areas = convert_h5_string_array(session_file, session_file['electrodeInfo']['area'])
        
                if area == 'all':
                    dimension_inds = np.arange(len(electrode_areas))
                else:
                    # dimension_inds = np.where(electrode_info['area'] == area)[0]
                    dimension_inds = np.where([area in area_entry for area_entry in electrode_areas])[0]
                dimension_inds = np.array([d for d in dimension_inds if d not in noise_filter_info[session]['bad_electrodes']])

                # Get PCA explained variance ratios
                area_pca_file = os.path.join(pca_dir, area)
                if os.path.exists(area_pca_file):
                    log_msg(f"Found PCA info for {session}: {area}", log, verbose)
                    pca_explained_variance_ratios = pd.read_pickle(area_pca_file)
                else:
                    log_msg(f"Computing PCA info for {session}: {area}", log, verbose)
                    grid_search_window_start_ts = get_grid_search_window_ts(cfg, session, valid_window_starts=noise_filter_info[session]['valid_window_starts'], random_state=cfg.params.random_state)
        
                    dt = session_file['lfpSchema']['smpInterval'][0, 0]
                    lfp = TransposedDatasetView(session_file['lfp']).transpose()
                
                    pca_explained_variance_ratios = {}
                    for t in tqdm(grid_search_window_start_ts, desc=f'Doing PCA for {session}: {area}'):
                        lfp_window = lfp[int(t/dt):int((t + cfg.params.window)/dt), dimension_inds]
                        lfp_window = lfp_window[::cfg.params.subsample]
                        # filter
                        lfp_window = filter_data(lfp_window, cfg.params.low_pass, cfg.params.high_pass, dt*cfg.params.subsample)
                        
                        pca = PCA(n_components=lfp_window.shape[1]).fit(lfp_window)
                        pca_explained_variance_ratios[t] = pca.explained_variance_ratio_
        
                    pd.to_pickle(pca_explained_variance_ratios, area_pca_file)
        
                # pick the max number of PCs across windows greater than the threshold
                pca_chosen[session][area] = np.max([np.argmax(np.cumsum(evr) >= cfg.params.pca_thresh) for evr in pca_explained_variance_ratios.values()])
    else:
        log_msg("Nevermind, not running PCA - PCA flag is False", log, verbose)
        pca_chosen = None
    
    return pca_chosen

def collect_grid_indices_to_run(cfg, session_list, areas, noise_filter_info, pca_chosen, log=None, verbose=False):
    # unfinished_sessions = False
    all_indices_to_run = {}
    for session in tqdm(session_list, desc='Getting Grid Search Run List', disable=not verbose):
        grid_search_window_start_ts = get_grid_search_window_ts(cfg, session, valid_window_starts=noise_filter_info[session]['valid_window_starts'], random_state=cfg.params.random_state)
        normed_folder = 'NOT_NORMED' if not cfg.params.normed else 'NORMED'
        filter_folder = f"[{cfg.params.high_pass},{cfg.params.low_pass}]" if cfg.params.low_pass is not None or cfg.params.high_pass is not None else 'NO_FILTER'
        grid_search_run_list = get_grid_search_run_list(cfg, session, grid_search_window_start_ts, noise_filter_info[session]['bad_electrodes'], verbose=verbose)
        noise_filter_folder = f"NOISE_FILTERED_{cfg.params.window}_{cfg.params.wake_amplitude_thresh}_{cfg.params.anesthesia_amplitude_thresh}_{cfg.params.electrode_num_thresh}" if cfg.params.noise_filter else "NO_NOISE_FILTER"
    
        for area in areas:
            
            pca_folder = "NO_PCA" if not cfg.params.pca else f"PCA_{pca_chosen[session][area]}"
            save_dir = os.path.join(cfg.params.grid_search_results_dir, cfg.params.data_class, 'grid_search_results', session, noise_filter_folder, normed_folder, f"SUBSAMPLE_{cfg.params.subsample}", filter_folder, f"WINDOW_{cfg.params.window}", cfg.params.grid_set, area, pca_folder)

            os.makedirs(save_dir, exist_ok=True)
            
            saved_files = os.listdir(save_dir)

            if not cfg.params.group_ranks:
                # filter runs for those with valid rank
                filtered_run_list = []
                for run_index, run_info in enumerate(grid_search_run_list[area]):
                    if cfg.params.pca:
                        n_dims = pca_chosen[session][area]
                    else:
                        n_dims = len(run_info['dimension_inds'])
                        if run_info['n_delays']*n_dims >= run_info['rank']:
                            run_info['run_index'] = run_index
                            filtered_run_list.append(run_info)
            else: # runs don't need to be filtered
                filtered_run_list = []
                for run_index, run_info in enumerate(grid_search_run_list[area]):
                    run_info['run_index'] = run_index
                    filtered_run_list.append(run_info)
            
            indices_to_run = []
            for run_info in filtered_run_list:
                filename = f"run_index-{run_info['run_index']}.pkl"
                if filename not in saved_files:
                    indices_to_run.append(run_info['run_index'])
        
            if len(indices_to_run) == 0:
                log_msg(f"*COMPLETE*: All results completed for {session} - {area}!!", log, verbose)
            elif len(indices_to_run) == len(filtered_run_list):
                log_msg(f"NOT STARTED: no results completed for {session} - {area}. Adding all indices! ({len(indices_to_run)})", log, verbose)
                if session not in all_indices_to_run:
                    all_indices_to_run[session] = {}
                all_indices_to_run[session][area] = indices_to_run
                # unfinished_sessions = True
            else:
                log_msg(f"INCOMPLETE: {session} - {area} incomplete, adding indices {indices_to_run}", log, verbose)
                if session not in all_indices_to_run:
                    all_indices_to_run[session] = {}
                all_indices_to_run[session][area] = indices_to_run
                # unfinished_sessions = True
    return all_indices_to_run

def get_grid_params_to_use(cfg, session_list, areas, pca_chosen, log=None, verbose=False):
    section_info, _, _, _ = get_section_info(session_list[0], cfg.params.all_data_dir, cfg.params.data_class)
    grid_search_results = get_grid_search_results(cfg, session_list, areas, len(section_info), pca_chosen, verbose=verbose)

    # sections_to_use = ['awake oddball', 'unconscious oddball', 'recovery oddball']
    # grid_search_results = resection_grid_results(cfg, grid_search_results, cfg.params.sections_to_use)

    for session in session_list:
        for area in areas:
            n_delays = grid_search_results[session][area]['n_delays']
            rank = grid_search_results[session][area]['rank']
            log_msg(f"Results for session: {session}, area: {area}: n_delays={n_delays}, rank={rank}", log, verbose)
    
    grid_params_to_use = {}
    for session in session_list:
        grid_params_to_use[session] = {}
        for area in areas:
            grid_params_to_use[session][area] = {'n_delays': grid_search_results[session][area]['n_delays'], 'rank': grid_search_results[session][area]['rank']}
    return grid_params_to_use

def collect_delase_indices_to_run(cfg, session_list, areas, noise_filter_info, pca_chosen,grid_params_to_use, log=None, verbose=False):
    all_indices_to_run = {}
    for session in tqdm(session_list, desc='Getting DeLASE Run List', disable=not verbose):
        delase_run_list = get_delase_run_list(cfg, session, valid_window_starts=noise_filter_info[session]['valid_window_starts'], bad_electrodes=noise_filter_info[session]['bad_electrodes'], verbose=verbose)

        noise_filter_folder = f"NOISE_FILTERED_{cfg.params.window}_{cfg.params.wake_amplitude_thresh}_{cfg.params.anesthesia_amplitude_thresh}_{cfg.params.electrode_num_thresh}" if cfg.params.noise_filter else "NO_NOISE_FILTER"
        normed_folder = 'NOT_NORMED' if not cfg.params.normed else 'NORMED'
        filter_folder = f"[{cfg.params.high_pass},{cfg.params.low_pass}]" if cfg.params.low_pass is not None or cfg.params.high_pass is not None else 'NO_FILTER'
        # sections_to_use_folder = 'SECTIONS_TO_USE_' + '__'.join(['_'.join(section.split(' ')) for section in cfg.params.sections_to_use])

        for area in areas:
            grid_folder = f"GRID_RESULTS_n_delays_{grid_params_to_use[session][area]['n_delays']}_rank_{grid_params_to_use[session][area]['rank']}"
            pca_folder = "NO_PCA" if not cfg.params.pca else f"PCA_{pca_chosen[session][area]}"
            save_dir = os.path.join(cfg.params.delase_results_dir, cfg.params.data_class, 'delase_results', session, noise_filter_folder, normed_folder, f"SUBSAMPLE_{cfg.params.subsample}", filter_folder, f"WINDOW_{cfg.params.window}", grid_folder, f"STRIDE_{cfg.params.stride}", area, pca_folder)

            os.makedirs(save_dir, exist_ok=True)
            
            saved_files = os.listdir(save_dir)
            indices_to_run = []
            for run_index in range(len(delase_run_list[area])):
                filename = f"run_index-{run_index}.pkl"
                if filename not in saved_files:
                    indices_to_run.append(run_index)
            
            if len(indices_to_run) == 0:
                log_msg(f"*COMPLETE*: All results completed for {session} - {area}!!", log, verbose)
            elif len(indices_to_run) == len(delase_run_list[area]):
                log_msg(f"NOT STARTED: no results completed for {session} - {area}. Adding all indices! ({len(indices_to_run)})", log, verbose)
                if session not in all_indices_to_run:
                    all_indices_to_run[session] = {}
                all_indices_to_run[session][area] = indices_to_run
            else:
                log_msg(f"INCOMPLETE: {session} - {area} incomplete, adding indices {indices_to_run}", log, verbose)
                if session not in all_indices_to_run:
                    all_indices_to_run[session] = {}
                all_indices_to_run[session][area] = indices_to_run
    return all_indices_to_run