#!/bin/bash
#SBATCH --job-name=delaseq
#SBATCH --partition=ou_bcs_low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/home/eisenaj/code/UniversalUnconsciousness/_slurm/logs/delaseq_%j.log
#SBATCH --error=/home/eisenaj/code/UniversalUnconsciousness/_slurm/logs/delaseq_%j.err

source activate universal-unconsciousness

python /home/eisenaj/code/UniversalUnconsciousness/UniversalUnconsciousness/DeLASE_analysis/delase_queueing_script.py