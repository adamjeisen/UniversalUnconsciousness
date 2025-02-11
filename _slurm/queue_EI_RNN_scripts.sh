#!/bin/bash
#SBATCH --job-name=eirnnq
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --exclude=node[001-079],node091
#SBATCH --output=/om2/user/eisenaj/code/UniversalUnconsciousness/_slurm/logs/eirnnq_%j.log
#SBATCH --error=/om2/user/eisenaj/code/UniversalUnconsciousness/_slurm/logs/eirnnq_%j.err

source activate universal-unconsciousness

# python /home/eisenaj/code/UniversalUnconsciousness/UniversalUnconsciousness/DeLASE_analysis/delase_queueing_script.py
HYDRA_FULL_ERROR=1 python /om2/user/eisenaj/code/UniversalUnconsciousness/UniversalUnconsciousness/network_modelling/EI_model_queue.py
