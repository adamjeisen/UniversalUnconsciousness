from copy import deepcopy
from delase import DeLASE
from delase.metrics import mase
import h5py
import hydra
import logging
from matplotlib import font_manager
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.stats import mannwhitneyu, pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import sys
from tqdm.auto import tqdm

# Disable HDF5 file locking globally for this queueing script
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from UniversalUnconsciousness.data_utils import filter_data, find_noisy_data, get_delase_results, get_delase_run_list, get_grid_search_results, get_grid_search_run_list, get_grid_search_window_ts, get_section_info, load_session_data, load_window_from_chunks, resection_grid_results
from UniversalUnconsciousness.data_utils import collect_delase_hyperparam_indices_to_run, collect_grid_indices_to_run, get_grid_params_to_use, get_grid_search_run_list, get_noise_filter_info, get_pca_chosen
# from UniversalUnconsciousness.plot_utils import load_font
# plt.style.use('UniversalUnconsciousness.sci_style')
# load_font()

log = logging.getLogger('DeLASE Hyperparam Queueing Script')

@hydra.main(config_path="conf", config_name="config.yaml", version_base='1.3')
def main(cfg):
    # --------------------------------------------------------------------------
    # Load session list
    # --------------------------------------------------------------------------
    # if os.path.exists('/om2/user/eisenaj'):
    #     cfg.params.all_data_dir = cfg.params.all_data_dir_openmind
    #     cfg.params.results_dir = cfg.params.results_dir_openmind
    # else:
    #     cfg.params.all_data_dir = cfg.params.all_data_dir_engaging
    #     cfg.params.results_dir = cfg.params.results_dir_engaging

    if 'propofol' in cfg.params.data_class:
        session_list = [f[:-4] for f in os.listdir(os.path.join(cfg.params.all_data_dir, 'anesthesia', 'mat', cfg.params.data_class)) if f.endswith('.mat')]
    else:
        session_list = [f[:-4] for f in os.listdir(os.path.join(cfg.params.all_data_dir, cfg.params.data_class, 'mat')) if f.endswith('.mat')]
    session_list = [session for session in session_list if session not in ['PEDRI_Ketamine_20220203']]

    # session_list = [session for session in session_list if 'Dex' not in session]
    # session_list = [session for session in session_list if 'Dex' in session]

    # get only high dose sessions
    if 'propofol' not in cfg.params.data_class:
        high_dose_session_list = []
        for session in session_list:
            if 'propofol' in cfg.params.data_class:
                session_file = h5py.File(os.path.join(cfg.params.all_data_dir, 'anesthesia', 'mat', cfg.params.data_class, session + '.mat'), 'r')
            else:
                session_file = h5py.File(os.path.join(cfg.params.all_data_dir, cfg.params.data_class, 'mat', session + '.mat'), 'r')
            dose = session_file['sessionInfo']['dose'][0, 0]
            if dose > 9:
                high_dose_session_list.append(session)
        session_list = high_dose_session_list

    # areas = ['all']
    areas = cfg.params.areas

    # session_list = [session_list[0]]

    # session_list = session_list[:1]
    log.info(f"Session list: {session_list}")

    # --------------------------------------------------------------------------
    # Find noisy data
    # --------------------------------------------------------------------------
    log.info("-"*20)
    log.info("FINDING NOISY DATA")
    log.info("-"*20)
    noise_filter_info = get_noise_filter_info(cfg, session_list, log=log, verbose=True)
    
    # --------------------------------------------------------------------------
    # PCA
    # --------------------------------------------------------------------------
    log.info("-"*20)
    log.info("RUNNING PCA")
    log.info("-"*20)
    pca_chosen = get_pca_chosen(cfg, session_list, areas, noise_filter_info, log=log, verbose=True)

    # --------------------------------------------------------------------------
    # DeLASE Across Hyperparameters
    # --------------------------------------------------------------------------
    loop_num = 1
    n_delays_vals = cfg.grid_sets[cfg.params.grid_set].n_delays_vals
    rank_vals = cfg.grid_sets[cfg.params.grid_set].rank_vals
    while True:
        log.info(f"DELASE PROCESS - LOOP #{loop_num}")
        # --------------------------------------------------------------------------
        # Collect indices to run
        # --------------------------------------------------------------------------
        log.info("-"*20)
        log.info("COLLECTING INDICES TO RUN")
        log.info("-"*20)
        all_indices_to_run = collect_delase_hyperparam_indices_to_run(cfg, session_list, areas, noise_filter_info, pca_chosen, n_delays_vals, rank_vals, log=log, verbose=True)
        
        # --------------------------------------------------------------------------
        # Running DeLASE Across Hyperparameters
        # --------------------------------------------------------------------------
        if os.path.exists('/om2/user/eisenaj'):
            os.chdir('/om2/user/eisenaj/code/UniversalUnconsciousness/UniversalUnconsciousness')
            batch_size = 250
            # batch_size = 1
        else:
            os.chdir('/home/eisenaj/code/UniversalUnconsciousness/UniversalUnconsciousness')
            batch_size = 25
        # batch_size = cfg.params.batch_size

        if not all_indices_to_run:
            log.info("No indices to run - breaking")
            break

        sessions_to_run = list(all_indices_to_run.keys())
        # sessions_to_run = sessions_to_run[:4]

        # iterator = tqdm(total=np.sum([len(all_indices_to_run[session]) for session in sessions_to_run]), desc='Hydra Multiprocessing - DSA on Neural Data')
        total_its = 0
        for session in sessions_to_run:
            for n_delays in all_indices_to_run[session].keys():
                for area in all_indices_to_run[session][n_delays].keys():
                    total_its += int(np.ceil(len(all_indices_to_run[session][n_delays][area])/batch_size))
        iterator = tqdm(total=total_its, desc='Hydra Multiprocessing - DeLASE on Neural Data')
        # iterator = tqdm(total=np.sum([np.sum([int(np.ceil(len(all_indices_to_run[session][n_delays][area])/batch_size)) for area in all_indices_to_run[session][n_delays].keys()]) for n_delays in all_indices_to_run[session].keys() for session in sessions_to_run]), desc='Hydra Multiprocessing - DeLASE on Neural Data')

        ranks = cfg.grid_sets[cfg.params.grid_set].rank_vals

        for session in sessions_to_run:
            for n_delays in all_indices_to_run[session].keys():
                for area in all_indices_to_run[session][n_delays].keys():
                    print(f"Running indices for {session} - {area} - n_delays {n_delays}")
                    num_batches = int(np.ceil(len(all_indices_to_run[session][n_delays][area])/batch_size))
                    for batch_num in range(num_batches):
                        batch_start = batch_num*batch_size
                        batch_end = np.min([(batch_num + 1)*batch_size, len(all_indices_to_run[session][n_delays][area])])
                        log.info(f"running batch #{batch_num}")
                        print(f"running batch #{batch_num}")
                        env_prefix = "HDF5_USE_FILE_LOCKING=FALSE HYDRA_FULL_ERROR=1"
                        if cfg.params.pca:
                            os.system(f"{env_prefix} python DeLASE_analysis/run_delase_across_hypers.py -m ++params.session={session} ++params.area={area} ++params.pca_dims={int(pca_chosen[session][area])} ++params.n_delays={n_delays} ++params.rank=[{','.join([str(int(r)) for r in ranks])}] ++params.run_index={','.join([str(i) for i in all_indices_to_run[session][n_delays][area][batch_start:batch_end]])}")
                        else:
                            os.system(f"{env_prefix} python DeLASE_analysis/run_delase_across_hypers.py -m ++params.session={session} ++params.area={area} ++params.n_delays={n_delays} ++params.rank=[{','.join([str(int(r)) for r in ranks])}] ++params.run_index={','.join([str(i) for i in all_indices_to_run[session][n_delays][area][batch_start:batch_end]])}")
                            # os.system(f"HYDRA_FULL_ERROR=1 python DeLASE_analysis/run_delase_across_hypers.py ++params.session={session} ++params.area={area} ++params.n_delays={n_delays} ++params.rank=[{','.join([str(int(r)) for r in ranks])}] ++params.run_index={','.join([str(i) for i in all_indices_to_run[session][n_delays][area][batch_start:batch_end]])}")
                        iterator.update()
                        # raise ValueError("Stop here")
        iterator.close()

        loop_num += 1

if __name__ == "__main__":
    main()