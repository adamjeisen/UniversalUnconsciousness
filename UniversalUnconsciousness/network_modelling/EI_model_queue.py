import hydra
from math import sqrt
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import torchdiffeq
from tqdm.auto import tqdm
from UniversalUnconsciousness.network_modelling.EI_RNN import EI_RNN, compute_lyaps_from_sol
from UniversalUnconsciousness.plot_utils import *
import yaml

@hydra.main(config_path="conf", config_name="config.yaml", version_base='1.3')
def main(cfg):
    os.chdir('/om2/user/eisenaj/code/UniversalUnconsciousness')
    g_vals = [1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65]
    ketamine_scales = [0.9, 0.95, 0.99, 0.999, 0.9999, 1, 1.0001, 1.001, 1.01, 1.05, 1.1]
    ketamine_doses = ['low', 'high']

    param_combinations = [(g, ks, kd) 
                        for g in g_vals 
                        for ks in ketamine_scales 
                        for kd in ketamine_doses]

    param_folder = f"N_{cfg.params.EI_RNN_params.N}__K_{cfg.params.EI_RNN_params.K}__alpha_{cfg.params.EI_RNN_params.alpha}__m0_{cfg.params.EI_RNN_params.m_0}__tau_{cfg.params.EI_RNN_params.tau}"
    NUM_SIMS = 10
    for num_sim in range(NUM_SIMS):
        random_state = 42 + num_sim
        while True:
            combos_to_run = []
            for (g, ks, kd) in param_combinations:
                save_file = os.path.join(cfg.params.save_dir, param_folder, f'KET_RS_{random_state}_g_{g}_ks_{ks}_kd_{kd}.pkl')
                if not os.path.exists(save_file):
                    combos_to_run.append((g, ks, kd))
            print(f"Running {len(combos_to_run)} combinations")
            if len(combos_to_run) > 0:
                # Take first 2 combinations to run
                
                # Read existing config
                config_path = "UniversalUnconsciousness/network_modelling/conf/config.yaml"
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Update the list_params section
                config['hydra']['sweeper']['list_params'] = {
                    'params.EI_RNN_params.g': [str(g) for g, _, _ in combos_to_run],
                    'params.EI_RNN_params.ketamine_scale': [str(ks) for _, ks, _ in combos_to_run],
                    'params.EI_RNN_params.ketamine_dose': [str(kd) for _, _, kd in combos_to_run]
                }
                
                # Write updated config
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                # Run the command
                cmd = f"python UniversalUnconsciousness/network_modelling/run_EI_model.py -m ++params.random_state={random_state}"
                os.system(cmd)
            else:
                print("No more combos to run")
                break

if __name__ == "__main__":
    main()


