from DSA.dsa import DSA
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
log = logging.getLogger('DSA Logger')
log.info("Current directory: " + os.getcwd())
from data_utils import get_data_class, get_dsa_run_list, load_session_data, load_window_from_chunks


def compute_dsa(cfg, session, run_params):
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    session_vars, T, N, dt = load_session_data(session, cfg.params.all_data_dir, ['lfpSchema'], data_class=get_data_class(session, cfg.params.all_data_dir))

    directory = pd.read_pickle(run_params['directory_path'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    result = {}

    lfp1 = load_window_from_chunks(run_params['window1_start'], run_params['window1_end'], directory, dimension_inds=run_params['dimension_inds'])
    lfp1 = lfp1[::cfg.params.subsample]
    lfp2 = load_window_from_chunks(run_params['window2_start'], run_params['window2_end'], directory, dimension_inds=run_params['dimension_inds'])
    lfp2 = lfp2[::cfg.params.subsample]
    if cfg.params.pca_dims is not None:
        pca1 = PCA(n_components=cfg.params.pca_dims)
        lfp1 = pca1.fit_transform(lfp1)
        pca2 = PCA(n_components=cfg.params.pca_dims)
        lfp2 = pca2.fit_transform(lfp2)
    

    n_delays = cfg.params.n_delays
    rank = cfg.params.rank

    # --------------------
    # FIT DELASE
    # --------------------

    log.info("Fitting DSA")
    start = time.time()

    dsa = DSA(lfp1, lfp2, lr=1e-3, iters=3000, n_delays=n_delays, rank=rank, device=device, verbose=True)
    score = dsa.fit_score()
    log.info(f"DSA fit in {time.time() - start} seconds")

    # if this is the first time we're seeing this window, test the predictivity
    if run_params['i'] == 0:
        if run_params['j'] == 1:
            log.info(f"Fitting metrics for window i = {run_params['i']}")
            lfp1_test = load_window_from_chunks(run_params['test_window1_start'], run_params['test_window1_end'], directory, dimension_inds=run_params['dimension_inds'])
            lfp1_test = lfp1_test[::cfg.params.subsample]

            if cfg.params.pca_dims is not None:
                lfp1_test = pca1.transform(lfp1_test)

            dsa.dmds[0][0].all_to_device(device)
            lfp1_test_pred = dsa.dmds[0][0].predict(test_data=torch.from_numpy(lfp1_test).to(device)).cpu()

            # collect results
            result = result | dict(metrics_i=dict(
                mase=mase(lfp1_test, lfp1_test_pred),
                r2=r2(lfp1_test, lfp1_test_pred),
                aic=aic(lfp1_test, lfp1_test_pred, rank=rank),
                mse=mse(lfp1_test, lfp1_test_pred)
            ))
        
        log.info(f"Fitting metrics for window j = {run_params['j']}")
        lfp2_test = load_window_from_chunks(run_params['test_window2_start'], run_params['test_window2_end'], directory, dimension_inds=run_params['dimension_inds'])
        lfp2_test = lfp2_test[::cfg.params.subsample]

        if cfg.params.pca_dims is not None:
                lfp2_test = pca2.transform(lfp2_test)

        dsa.dmds[1][0].all_to_device(device)
        lfp2_test_pred = dsa.dmds[1][0].predict(test_data=torch.from_numpy(lfp2_test).to(device)).cpu()

        # collect results
        result = result | dict(metrics_j=dict(
            mase=mase(lfp2_test, lfp2_test_pred),
            r2=r2(lfp2_test, lfp2_test_pred),
            aic=aic(lfp2_test, lfp2_test_pred, rank=rank),
            mse=mse(lfp2_test, lfp2_test_pred)
        ))

    # collect results

    result = result | dict(
        score=score,
    )

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
    dsa_run_list = get_dsa_run_list(cfg, cfg.params.session)
    run_params = dsa_run_list[cfg.params.area][cfg.params.run_index]

    normed_folder = 'NOT_NORMED' if not cfg.params.normed else 'NORMED'
    filter_folder = f"[{cfg.params.high_pass},{cfg.params.low_pass}]" if cfg.params.low_pass is not None or cfg.params.high_pass is not None else 'NO_FILTER'
    pca_folder = "NO_PCA" if not cfg.params.pca else f"PCA_{cfg.params.pca_dims}"
    save_dir = os.path.join(cfg.params.dsa_results_dir, cfg.params.data_class, 'dsa_results', cfg.params.session, normed_folder, f"SUBSAMPLE_{cfg.params.subsample}", filter_folder, f"WINDOW_{cfg.params.window}", cfg.params.grid_set, f"STAT_TO_USE_{cfg.params.stat_to_use}", f"STRIDE_{cfg.params.stride}", run_params['area'], pca_folder)
    os.makedirs(save_dir, exist_ok=True)

    save_file_path = os.path.join(save_dir, f"run_index-{cfg.params.run_index}.pkl")

    if os.path.exists(save_file_path):
        print("skip")
        log.info(f"Session {cfg.params.session} area {cfg.params.area} run index {cfg.params.run_index} already exists. Skipping.")
    else:
        log.info(f"Session {cfg.params.session} area {cfg.params.area} run index {cfg.params.run_index} does not exist. Running.")

        session = cfg.params.session
        result = compute_dsa(cfg, session, run_params)

        log.info("Saving results")
        pd.to_pickle(result, save_file_path)
        

if __name__ == '__main__':
    main()