import h5py
import hydra
from hydra import compose, initialize
import logging
from matplotlib import font_manager
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
import sys
from tqdm.auto import tqdm

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from UniversalUnconsciousness.data_utils import collect_delase_indices_to_run, collect_grid_indices_to_run, get_grid_params_to_use, get_grid_search_run_list, get_noise_filter_info, get_pca_chosen
from UniversalUnconsciousness.hdf5_utils import convert_char_array, convert_h5_string_array, TransposedDatasetView

log = logging.getLogger('DeLASE Queueing Script')

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

    areas = ['all']

    # session_list = [session_list[0]]

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
    # Grid Search
    # --------------------------------------------------------------------------

    loop_num = 1
    while True:
        log.info(f"GRID SEARCH PROCESS - LOOP #{loop_num}")
        # --------------------------------------------------------------------------
        # Collect indices to run
        # --------------------------------------------------------------------------
        log.info("-"*20)
        log.info("COLLECTING INDICES TO RUN")
        log.info("-"*20)
        
        all_indices_to_run = collect_grid_indices_to_run(cfg, session_list, areas, noise_filter_info, pca_chosen, log=log, verbose=True)
        # --------------------------------------------------------------------------
        # Running grid search
        # --------------------------------------------------------------------------
        if os.path.exists('/om2/user/eisenaj'):
            os.chdir('/om2/user/eisenaj/code/UniversalUnconsciousness/UniversalUnconsciousness')
            batch_size = 250
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
        iterator = tqdm(total=np.sum([np.sum([int(np.ceil(len(all_indices_to_run[session][area])/batch_size)) for area in all_indices_to_run[session].keys()]) for session in sessions_to_run]), desc='Hydra Multiprocessing - Grid Search on Neural Data')

        for session in sessions_to_run:
            for area in all_indices_to_run[session].keys():
                log.info(f"Running indices for {session} - {area}")
                num_batches = int(np.ceil(len(all_indices_to_run[session][area])/batch_size))
                for batch_num in range(num_batches):
                    batch_start = batch_num*batch_size
                    batch_end = np.min([(batch_num + 1)*batch_size, len(all_indices_to_run[session][area])])
                    log.info(f"running batch #{batch_num}")
                    if cfg.params.pca:
                        os.system(f"HYDRA_FULL_ERROR=1 python DeLASE_analysis/run_grid_search.py -m ++params.session={session} ++params.area={area} ++params.pca_dims={int(pca_chosen[session][area])} ++params.run_index={','.join([str(i) for i in all_indices_to_run[session][area][batch_start:batch_end]])}")
                    else:
                        os.system(f"HYDRA_FULL_ERROR=1 python DeLASE_analysis/run_grid_search.py -m ++params.session={session} ++params.area={area} ++params.run_index={','.join([str(i) for i in all_indices_to_run[session][area][batch_start:batch_end]])}")
                    iterator.update()   
        iterator.close()

        loop_num += 1
    
    # --------------------------------------------------------------------------
    # Collect grid search results
    # --------------------------------------------------------------------------
    cfg.params.stat_to_use = 'aic'
    
    grid_params_to_use = get_grid_params_to_use(cfg, session_list, areas, pca_chosen, log=log, verbose=True)

    # --------------------------------------------------------------------------
    # DeLASE
    # --------------------------------------------------------------------------
    loop_num = 1
    while True:
        log.info(f"DELASE PROCESS - LOOP #{loop_num}")
        # --------------------------------------------------------------------------
        # Collect indices to run
        # --------------------------------------------------------------------------
        log.info("-"*20)
        log.info("COLLECTING INDICES TO RUN")
        log.info("-"*20)
        all_indices_to_run = collect_delase_indices_to_run(cfg, session_list, areas, noise_filter_info, pca_chosen, grid_params_to_use, log=log, verbose=True)
        
        # --------------------------------------------------------------------------
        # Running DeLASE
        # --------------------------------------------------------------------------
        if os.path.exists('/om2/user/eisenaj'):
            os.chdir('/om2/user/eisenaj/code/UniversalUnconsciousness/UniversalUnconsciousness')
            batch_size = 250
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
        iterator = tqdm(total=np.sum([np.sum([int(np.ceil(len(all_indices_to_run[session][area])/batch_size)) for area in all_indices_to_run[session].keys()]) for session in sessions_to_run]), desc='Hydra Multiprocessing - DeLASE on Neural Data')

        for session in sessions_to_run:
            for area in all_indices_to_run[session].keys():
                print(f"Running indices for {session} - {area}")
                num_batches = int(np.ceil(len(all_indices_to_run[session][area])/batch_size))
                for batch_num in range(num_batches):
                    batch_start = batch_num*batch_size
                    batch_end = np.min([(batch_num + 1)*batch_size, len(all_indices_to_run[session][area])])
                    print(f"running batch #{batch_num}")
                    if cfg.params.pca:
                        os.system(f"HYDRA_FULL_ERROR=1 python DeLASE_analysis/run_delase.py -m ++params.session={session} ++params.area={area} ++params.pca_dims={int(pca_chosen[session][area])} ++params.n_delays={int(grid_search_results[session][area]['n_delays'])} ++params.rank={int(grid_search_results[session][area]['rank'])} ++params.run_index={','.join([str(i) for i in all_indices_to_run[session][area][batch_start:batch_end]])}")
                    else:
                        os.system(f"HYDRA_FULL_ERROR=1 python DeLASE_analysis/run_delase.py -m ++params.session={session} ++params.area={area} ++params.n_delays={int(grid_params_to_use[session][area]['n_delays'])} ++params.rank={int(grid_params_to_use[session][area]['rank'])} ++params.run_index={','.join([str(i) for i in all_indices_to_run[session][area][batch_start:batch_end]])}")
                    iterator.update()
        iterator.close()

        loop_num += 1

if __name__ == "__main__":
    main()