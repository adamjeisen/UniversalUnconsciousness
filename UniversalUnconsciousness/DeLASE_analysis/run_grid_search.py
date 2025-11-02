import h5py
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

from delase import DeLASE
from delase.dmd import DMD
from delase.metrics import aic, mase, mse, r2_score

log = logging.getLogger('Grid Search Logger')
log.info("Current directory: " + os.getcwd())
from UniversalUnconsciousness.data_utils import filter_data, get_data_class, get_grid_search_run_list, load_session_data, load_window_from_chunks
from UniversalUnconsciousness.hdf5_utils import TransposedDatasetView

def compute_havok_fit(cfg, run_params):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if 'propofol' in cfg.params.data_class:
        session_file = h5py.File(os.path.join(cfg.params.all_data_dir, 'anesthesia', 'mat', cfg.params.data_class, cfg.params.session + '.mat'), 'r')
    else:
        session_file = h5py.File(os.path.join(cfg.params.all_data_dir, cfg.params.data_class, 'mat', cfg.params.session + '.mat'), 'r')
    dt = session_file['lfpSchema']['smpInterval'][0, 0]
    lfp_file = TransposedDatasetView(session_file['lfp']).transpose()
    lfp = lfp_file[int(run_params['window_start']/dt):int(run_params['window_end']/dt), run_params['dimension_inds']]
    if cfg.params.normed:
        lfp = (lfp - lfp.mean())/lfp.std()
    lfp = lfp[::cfg.params.subsample]
    lfp_test = lfp_file[int(run_params['test_window_start']/dt):int(run_params['test_window_end']/dt), run_params['dimension_inds']]
    lfp_test = lfp_test[::cfg.params.subsample]
    if cfg.params.normed:
        lfp_test = (lfp_test - lfp_test.mean())/lfp_test.std()
    
    lfp = filter_data(lfp, cfg.params.low_pass, cfg.params.high_pass, dt*cfg.params.subsample)
    lfp_test = filter_data(lfp_test, cfg.params.low_pass, cfg.params.high_pass, dt*cfg.params.subsample)

    log.info(f"LFP shape: {lfp.shape}")

    if cfg.params.pca is not None:
        pca = PCA(n_components=cfg.params.pca_dims)
        lfp = pca.fit_transform(lfp)
        lfp_test = pca.transform(lfp_test)

    # --------------------
    # FIT DMD
    # --------------------

    if cfg.params.compute_delase_on_grid:
        method = 'DeLASE'
    else:
        method = 'DMD'

    log.info(f"Fitting {method}")
    start = time.time()

    if isinstance(run_params['rank'], int):
        if rank <= lfp.shape[1]*run_params['n_delays']:
            if cfg.params.compute_delase_on_grid:
                delase = DeLASE(
                            lfp,
                            n_delays=run_params['n_delays'],
                            rank=run_params['rank'],
                            device=device,
                            dt=dt*cfg.params.subsample,
                            max_freq=cfg.params.max_freq,
                            max_unstable_freq=cfg.params.max_unstable_freq, 
                            verbose=True
                        )
                delase.fit()
            else:
                dmd = DMD(lfp, n_delays=run_params['n_delays'], rank=run_params['rank'], device=device, verbose=True)
                dmd.fit()
            log.info(f"{method} fit in {time.time() - start} seconds")

            n_delays = run_params['n_delays']
            rank = run_params['rank']
            if cfg.params.compute_delase_on_grid:
                lfp_test_pred = delase.DMD.predict(test_data=lfp_test).cpu()
            else:
                lfp_test_pred = dmd.predict(test_data=lfp_test).cpu()

            # collect results
            result = dict(
                mase=mase(lfp_test[n_delays:], lfp_test_pred[n_delays:].cpu().numpy()),
                r2=r2_score(lfp_test[n_delays:], lfp_test_pred[n_delays:].cpu().numpy()),
                aic=aic(lfp_test[n_delays:], lfp_test_pred[n_delays:].cpu().numpy(), k=rank**2),
                mse=mse(lfp_test[n_delays:], lfp_test_pred[n_delays:].cpu().numpy())
            )
            if cfg.params.compute_delase_on_grid:
                result['stability_params'] = delase.stability_params.cpu().numpy()
                result['stability_freqs'] = delase.stability_freqs.cpu().numpy()

        else:
            result = dict(
                mase=np.nan,
                r2=np.nan,
                aic=np.nan,
                mse=np.nan
            )
            if cfg.params.compute_delase_on_grid:
                result['stability_params'] = np.nan
                result['stability_freqs'] = np.nan
    else: # list or np.array
        result = []

        if cfg.params.compute_delase_on_grid:
            delase = DeLASE(
                        lfp,
                        n_delays=run_params['n_delays'],
                        rank=None,
                        device=device,
                        dt=dt*cfg.params.subsample,
                        max_freq=cfg.params.max_freq,
                        max_unstable_freq=cfg.params.max_unstable_freq, 
                        verbose=True
                    )
            delase.DMD.compute_hankel()
            delase.DMD.compute_svd()
        else:
            dmd = DMD(lfp, n_delays=run_params['n_delays'], rank=None, device=device, verbose=True)
            dmd.compute_hankel()
            dmd.compute_svd()

        for rank in run_params['rank']:
            if rank <= lfp.shape[1]*run_params['n_delays']:
                if cfg.params.compute_delase_on_grid:
                    delase.DMD.rank = rank
                    delase.DMD.compute_havok_dmd()
                    lfp_test_pred = delase.DMD.predict(test_data=lfp_test).cpu()
                    delase.get_stability()
                else:
                    dmd.rank = rank
                    dmd.compute_havok_dmd()

                    lfp_test_pred = dmd.predict(test_data=lfp_test).cpu()

                n_delays = run_params['n_delays']
                # collect results
                result.append(dict(
                    mase=mase(lfp_test[n_delays:], lfp_test_pred[n_delays:].cpu().numpy()),
                    r2=r2_score(lfp_test[n_delays:], lfp_test_pred[n_delays:].cpu().numpy()),
                    aic=aic(lfp_test[n_delays:], lfp_test_pred[n_delays:].cpu().numpy(), k=rank**2),
                    mse=mse(lfp_test[n_delays:], lfp_test_pred[n_delays:].cpu().numpy())
                ))
                if cfg.params.compute_delase_on_grid:
                    result[-1]['stability_params'] = delase.stability_params.cpu().numpy()
                    result[-1]['stability_freqs'] = delase.stability_freqs.cpu().numpy()
            else:
                result.append(dict(
                    mase=np.nan,
                    r2=np.nan,
                    aic=np.nan,
                    mse=np.nan
                ))
                if cfg.params.compute_delase_on_grid:
                    result[-1]['stability_params'] = np.nan
                    result[-1]['stability_freqs'] = np.nan

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
    grid_search_run_list = get_grid_search_run_list(cfg, cfg.params.session, verbose=True)
    run_params = grid_search_run_list[cfg.params.area][cfg.params.run_index]

    normed_folder = 'NOT_NORMED' if not cfg.params.normed else 'NORMED'
    noise_filter_folder = f"NOISE_FILTERED_{cfg.params.window}_{cfg.params.wake_amplitude_thresh}_{cfg.params.anesthesia_amplitude_thresh}_{cfg.params.electrode_num_thresh}_stride_{cfg.params.stride}" if cfg.params.noise_filter else "NO_NOISE_FILTER"
    filter_folder = f"[{cfg.params.high_pass},{cfg.params.low_pass}]" if cfg.params.low_pass is not None or cfg.params.high_pass is not None else 'NO_FILTER'
    pca_folder = "NO_PCA" if not cfg.params.pca else f"PCA_{cfg.params.pca_dims}"
    save_dir = os.path.join(cfg.params.grid_search_results_dir, cfg.params.data_class, 'grid_search_results', cfg.params.session, noise_filter_folder, normed_folder, f"SUBSAMPLE_{cfg.params.subsample}", filter_folder, f"WINDOW_{cfg.params.window}", cfg.params.grid_set, run_params['area'], pca_folder)
    os.makedirs(save_dir, exist_ok=True)

    save_file_path = os.path.join(save_dir, f"run_index-{cfg.params.run_index}.pkl")
    if cfg.params.testing:
        return compute_havok_fit(cfg, run_params)
    else:
        if os.path.exists(save_file_path):
            print("skip")
            log.info(f"Session {cfg.params.session} area {cfg.params.area} run index {cfg.params.run_index} already exists. Skipping.")
        else:
            log.info(f"Session {cfg.params.session} area {cfg.params.area} run index {cfg.params.run_index} does not exist. Running.")

            result = compute_havok_fit(cfg, run_params)

            log.info("Saving results to " + save_file_path)
            pd.to_pickle(result, save_file_path)
        

if __name__ == '__main__':
    main()