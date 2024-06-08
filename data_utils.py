from copy import deepcopy
import numpy as np
import os
import pandas as pd
from spynal.matIO import loadmat
import time

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
    
    filename = os.path.join(all_data_dir, data_class, f'{session}.mat')

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
        elif data_class == 'leverOddball':
            session_vars['electrodeInfo']['area'] = np.array([f"{area}-{h[0].upper()}" for area, h in zip(session_vars['electrodeInfo']['area'], session_vars['electrodeInfo']['hemisphere'])])
    T = len(session_vars['lfpSchema']['index'][0])
    N = len(session_vars['lfpSchema']['index'][1])
    dt = session_vars['lfpSchema']['smpInterval'][0]

    return session_vars, T, N, dt

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