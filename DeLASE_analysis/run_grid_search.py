from DSA.dmd import DMD
from DSA.stats import aic, mase, mse, r2
import hydra
import logging
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
import submitit
import sys
import time
import torch

sys.path.append('/om2/user/eisenaj/code/UniversalUnconsciousness')
log = logging.getLogger('Grid Search Logger')
log.info("Current directory: " + os.getcwd())
from data_utils import get_grid_search_run_list, load_window_from_chunks


def compute_havok_fit(cfg, run_params):

    directory = pd.read_pickle(run_params['directory_path'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    lfp = load_window_from_chunks(run_params['window_start'], run_params['window_end'], directory, dimension_inds=run_params['dimension_inds'])
    lfp = lfp[::cfg.params.subsample]
    lfp_test = load_window_from_chunks(run_params['test_window_start'], run_params['test_window_end'], directory, dimension_inds=run_params['dimension_inds'])
    lfp_test = lfp_test[::cfg.params.subsample]
    if cfg.params.pca is not None:
        pca = PCA(n_components=cfg.params.pca_dims)
        lfp = pca.fit_transform(lfp)
        lfp_test = pca.transform(lfp_test)

    # --------------------
    # FIT DMD
    # --------------------

    log.info("Fitting DMD")
    start = time.time()

    if isinstance(run_params['rank'], int):
        dmd = DMD(lfp, n_delays=run_params['n_delays'], rank=run_params['rank'], device=device, verbose=True)
        dmd.fit()
        log.info(f"DMD fit in {time.time() - start} seconds")

        lfp_test_pred = dmd.predict(test_data=lfp_test).cpu()

        # collect results
        result = dict(
            mase=mase(lfp_test, lfp_test_pred),
            r2=r2(lfp_test, lfp_test_pred),
            aic=aic(lfp_test, lfp_test_pred, rank=run_params['rank']),
            mse=mse(lfp_test, lfp_test_pred)
        )

    else: # list or np.array
        result = []

        dmd = DMD(lfp, n_delays=run_params['n_delays'], rank=None, device=device, verbose=True)
        dmd.compute_hankel()
        dmd.compute_svd()

        for rank in run_params['rank']:
            if rank <= lfp.shape[1]*run_params['n_delays']:
                dmd.rank = rank
                dmd.compute_havok_dmd()

                lfp_test_pred = dmd.predict(test_data=lfp_test).cpu()

                # collect results
                result.append(dict(
                    mase=mase(lfp_test, lfp_test_pred),
                    r2=r2(lfp_test, lfp_test_pred),
                    aic=aic(lfp_test, lfp_test_pred, rank=rank),
                    mse=mse(lfp_test, lfp_test_pred)
                ))
            else:
                result.append(dict(
                    mase=np.nan,
                    r2=np.nan,
                    aic=np.nan,
                    mse=np.nan
                ))

        log.info(f"All DMDs fit in {time.time() - start} seconds")

    return result

@hydra.main(config_path="conf", config_name="config.yaml", version_base='1.3')
def main(cfg):
    # INITIALIZE
    try:
        env = submitit.JobEnvironment()
        log.info(f"Process ID {os.getpid()} executing task {cfg.params.session}, {cfg.params.area}, {cfg.params.run_index}, with {env}")
    except RuntimeError as e:
        # print(e)
        log.info(f"Process ID {os.getpid()} executing task {cfg.params.session}, {cfg.params.area}, {cfg.params.run_index} locally")

    # GET RUN PARAMETERS
    print(cfg.params.session)
    grid_search_run_list = get_grid_search_run_list(cfg.params.session, os.path.join(cfg.params.grid_search_results_dir, cfg.params.data_class), cfg.params.all_data_dir, cfg.params.window, cfg.grid_sets[cfg.params.grid_set], cfg.params.grid_set, T_pred=cfg.params.T_pred, group_ranks=cfg.params.group_ranks, verbose=True, random_state=cfg.params.random_state)
    run_params = grid_search_run_list[cfg.params.area][cfg.params.run_index]

    normed_folder = 'NOT_NORMED' if not cfg.params.normed else 'NORMED'
    filter_folder = f"[{cfg.params.high_pass},{cfg.params.low_pass}]" if cfg.params.low_pass is not None or cfg.params.high_pass is not None else 'NO_FILTER'
    pca_folder = "NO_PCA" if not cfg.params.pca else f"PCA_{cfg.params.pca_dims}"
    save_dir = os.path.join(cfg.params.grid_search_results_dir, cfg.params.data_class, 'grid_search_results', cfg.params.session, normed_folder, f"SUBSAMPLE_{cfg.params.subsample}", filter_folder, f"WINDOW_{cfg.params.window}", cfg.params.grid_set, run_params['area'], pca_folder)
    os.makedirs(save_dir, exist_ok=True)

    save_file_path = os.path.join(save_dir, f"run_index-{cfg.params.run_index}.pkl")

    if os.path.exists(save_file_path):
        print("skip")
        log.info(f"Session {cfg.params.session} area {cfg.params.area} run index {cfg.params.run_index} already exists. Skipping.")
    else:
        log.info(f"Session {cfg.params.session} area {cfg.params.area} run index {cfg.params.run_index} does not exist. Running.")

        result = compute_havok_fit(cfg, run_params)

        log.info("Saving results")
        pd.to_pickle(result, save_file_path)
        

if __name__ == '__main__':
    main()