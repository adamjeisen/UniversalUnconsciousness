import hydra
import logging
from omegaconf import OmegaConf
import os
import pandas as pd
import torch
import torchdiffeq
log = logging.getLogger(__name__)

from UniversalUnconsciousness.network_modelling.EI_RNN import compute_lyaps_from_sol, EI_RNN

@hydra.main(config_path="conf", config_name="config.yaml", version_base='1.3')
def main(cfg):
    param_folder = f"N_{cfg.params.EI_RNN_params.N}__K_{cfg.params.EI_RNN_params.K}__alpha_{cfg.params.EI_RNN_params.alpha}__m0_{cfg.params.EI_RNN_params.m_0}__tau_{cfg.params.EI_RNN_params.tau}"
    save_file = os.path.join(cfg.params.save_dir, param_folder, f'KET_RS_{cfg.params.random_state}_g_{cfg.params.EI_RNN_params.g}_ks_{cfg.params.EI_RNN_params.ketamine_scale}_kd_{cfg.params.EI_RNN_params.ketamine_dose}.pkl')
    if os.path.exists(save_file):
        log.info(f'File {save_file} already exists. Skipping.')
        return
    log.info(f'Running EI model with config: {cfg}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f'Using device: {device}')

    torch.manual_seed(cfg.params.random_state)
    model = hydra.utils.instantiate(cfg.params.EI_RNN_params)
    model.to(device)

    h_0 = torch.randn(2*cfg.params.EI_RNN_params.N, device=device)
    time_vals = torch.linspace(0, cfg.params.T, int(cfg.params.T / cfg.params.dt))

    sol = torchdiffeq.odeint(model.forward, h_0, time_vals.to(device), method='rk4')
    transient_idx = int(cfg.params.transient_pct * len(time_vals))

    lyaps = compute_lyaps_from_sol(sol[transient_idx:], model.jac, dt=cfg.params.dt, k=3, verbose=True)

    pd.to_pickle({'model_params': model.to('cpu').parameter_dict(), 'h_0': h_0.cpu(), 'lyaps': lyaps.cpu(), 'cfg': OmegaConf.to_container(cfg, resolve=True)}, save_file)


if __name__ == "__main__":
    main()