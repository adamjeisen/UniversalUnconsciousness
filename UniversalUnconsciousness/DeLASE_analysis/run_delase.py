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

sys.path.append('/home/eisenaj/code/UniversalUnconsciousness')
log = logging.getLogger('DeLASE Logger')
log.info("Current directory: " + os.getcwd())
from data_utils import filter_data, get_data_class, get_delase_run_list, load_session_data, load_window_from_chunks

from delase import DeLASE 
from delase.metrics import aic, mase, mse, r2_score

from hdf5_utils import TransposedDatasetView

def compute_delase(cfg, session, run_params):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if 'propofol' in cfg.params.data_class:
        session_file = h5py.File(os.path.join(cfg.params.all_data_dir, 'anesthesia', 'mat', cfg.params.data_class, cfg.params.session + '.mat'), 'r')
    else:
        session_file = h5py.File(os.path.join(cfg.params.all_data_dir, cfg.params.data_class, 'mat', cfg.params.session + '.mat'), 'r')
    dt = session_file['lfpSchema']['smpInterval'][0, 0]
    lfp_file = TransposedDatasetView(session_file['lfp']).transpose()
    lfp = lfp_file[int(run_params['window_start']/dt):int(run_params['window_end']/dt), run_params['dimension_inds']]
    lfp = lfp[::cfg.params.subsample]
    if cfg.params.normed:
        lfp = (lfp - lfp.mean())/lfp.std()
    lfp_test = lfp_file[int(run_params['test_window_start']/dt):int(run_params['test_window_end']/dt), run_params['dimension_inds']]
    lfp_test = lfp_test[::cfg.params.subsample]
    if cfg.params.normed:
        lfp_test = (lfp_test - lfp_test.mean())/lfp_test.std()

    lfp = filter_data(lfp, cfg.params.low_pass, cfg.params.high_pass, dt*cfg.params.subsample)
    lfp_test = filter_data(lfp_test, cfg.params.low_pass, cfg.params.high_pass, dt*cfg.params.subsample)
    log.info(f"LFP shape: {lfp.shape}")

    if cfg.params.pca_dims is not None:
        pca = PCA(n_components=cfg.params.pca_dims)
        lfp = pca.fit_transform(lfp)
        lfp_test = pca.fit_transform(lfp_test)

    # --------------------
    # FIT DELASE
    # --------------------

    log.info("Fitting DeLASE")
    start = time.time()

    delase = DeLASE(lfp, 
        n_delays=cfg.params.n_delays,
        rank=cfg.params.rank,
        dt=dt*cfg.params.subsample,
        max_freq=cfg.params.max_freq,
        max_unstable_freq=cfg.params.max_unstable_freq,
        device=device,
        verbose=True
        )
    delase.fit()

    result = {} | run_params
    result['stability_params'] = delase.stability_params.cpu().numpy()
    result['stability_freqs'] = delase.stability_freqs.cpu().numpy()
    if cfg.params.area == 'all':
        result['Js'] = delase.Js.cpu().numpy()
    
    log.info(f"DeLASE fit in {time.time() - start} seconds")
    
    # --------------------
    # COMPUTE METRICS
    # --------------------

    log.info("Computing metrics")

    # HAVOK
    preds = delase.DMD.predict(lfp_test)
    preds = preds.cpu().numpy()
    aic_val = aic(lfp_test[delase.n_delays:], preds[delase.n_delays:], k=delase.DMD.A_v.shape[0]*delase.DMD.A_v.shape[1])
    mase_val = mase(lfp_test[delase.n_delays:], preds[delase.n_delays:])
    mse_val = mse(lfp_test[delase.n_delays:], preds[delase.n_delays:])
    r2_val = r2_score(lfp_test[delase.n_delays:], preds[delase.n_delays:])

    # persistence baseline

    aic_val_pb = aic(lfp_test[1:], lfp_test[:-1], k=0)
    mase_val_pb = mase(lfp_test[1:], lfp_test[:-1])
    mse_val_pb = mse(lfp_test[1:], lfp_test[:-1])
    r2_val_pb = r2_score(lfp_test[1:], lfp_test[:-1])

    # VAR

    A = np.linalg.lstsq(lfp[:-1], lfp[1:], rcond=1e-13)[0].T

    preds_VAR = (A @ lfp_test[:-1].T).T
    preds_VAR = np.vstack((lfp_test[[0]], preds_VAR))

    aic_val_VAR = aic(lfp_test[1:], preds_VAR[1:], k=A.shape[0]*A.shape[1])
    mase_val_VAR = mase(lfp_test[1:], preds_VAR[1:])
    mse_val_VAR = mse(lfp_test[1:], preds_VAR[1:])
    r2_val_VAR = r2_score(lfp_test[1:], preds_VAR[1:])

    # VAR small

    small_window = 500
    lfp_small = lfp[:small_window]
    lfp_small_test = lfp[small_window:2*small_window]
    A_small = np.linalg.lstsq(lfp_small[:-1], lfp_small[1:], rcond=1e-13)[0].T

    preds_VAR_small = (A_small @ lfp_small_test[:-1].T).T
    preds_VAR_small = np.vstack((lfp_small_test[[0]], preds_VAR_small))

    aic_val_VAR_small = aic(lfp_small_test[1:], preds_VAR_small[1:], k=A_small.shape[0]*A_small.shape[1])
    mase_val_VAR_small = mase(lfp_small_test[1:], preds_VAR_small[1:])
    mse_val_VAR_small = mse(lfp_small_test[1:], preds_VAR_small[1:])
    r2_val_VAR_small = r2_score(lfp_small_test[1:], preds_VAR_small[1:])

    log.info(f"AIC: DeLASE: {aic_val}, PB: {aic_val_pb}, VAR: {aic_val_VAR}, VAR_small: {aic_val_VAR_small}")
    log.info("Metrics computed!")

    # collect results

    result = result | dict(
        aic_val=aic_val,
        mase_val=mase_val,
        mse_val=mse_val,
        r2_val=r2_val,
        aic_val_pb=aic_val_pb,
        mase_val_pb=mase_val_pb,
        mse_val_pb=mse_val_pb,
        r2_val_pb=r2_val_pb,
        aic_val_VAR=aic_val_VAR,
        mase_val_VAR=mase_val_VAR,
        mse_val_VAR=mse_val_VAR,
        r2_val_VAR=r2_val_VAR,
        aic_val_VAR_small=aic_val_VAR_small,
        mase_val_VAR_small=mase_val_VAR_small,
        mse_val_VAR_small=mse_val_VAR_small,
        r2_val_VAR_small=r2_val_VAR_small
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
    delase_run_list = get_delase_run_list(cfg, cfg.params.session)
    
    run_params = delase_run_list[cfg.params.area][cfg.params.run_index]

    noise_filter_folder = f"NOISE_FILTERED_{cfg.params.window}_{cfg.params.wake_amplitude_thresh}_{cfg.params.anesthesia_amplitude_thresh}_{cfg.params.electrode_num_thresh}" if cfg.params.noise_filter else "NO_NOISE_FILTER"
    normed_folder = 'NOT_NORMED' if not cfg.params.normed else 'NORMED'
    filter_folder = f"[{cfg.params.high_pass},{cfg.params.low_pass}]" if cfg.params.low_pass is not None or cfg.params.high_pass is not None else 'NO_FILTER'
    grid_folder = f"GRID_RESULTS_n_delays_{cfg.params.n_delays}_rank_{cfg.params.rank}"
    sections_to_use_folder = 'SECTIONS_TO_USE_' + '__'.join(['_'.join(section.split(' ')) for section in cfg.params.sections_to_use])
    pca_folder = "NO_PCA" if not cfg.params.pca else f"PCA_{cfg.params.pca_dims}"
    save_dir = os.path.join(cfg.params.delase_results_dir, cfg.params.data_class, 'delase_results', cfg.params.session, noise_filter_folder, normed_folder, f"SUBSAMPLE_{cfg.params.subsample}", filter_folder, f"WINDOW_{cfg.params.window}", grid_folder, f"STRIDE_{cfg.params.stride}", run_params['area'], pca_folder)
    os.makedirs(save_dir, exist_ok=True)

    save_file_path = os.path.join(save_dir, f"run_index-{cfg.params.run_index}.pkl")

    if os.path.exists(save_file_path):
        print("skip")
        log.info(f"Session {cfg.params.session} area {cfg.params.area} run index {cfg.params.run_index} already exists. Skipping.")
    else:
        log.info(f"Session {cfg.params.session} area {cfg.params.area} run index {cfg.params.run_index} does not exist. Running.")

        session = cfg.params.session
        result = compute_delase(cfg, session, run_params)

        log.info("Saving results")
        pd.to_pickle(result, save_file_path)
        

if __name__ == '__main__':
    main()