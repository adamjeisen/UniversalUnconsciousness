#!/bin/bash
#SBATCH --job-name=delaseq
##SBATCH --partition=normal
#SBATCH --partition=millerlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
##SBATCH --exclude=node[001-079],node091
#SBATCH --output=/om2/user/eisenaj/code/UniversalUnconsciousness/_slurm/logs/delaseq_%j.log
#SBATCH --error=/om2/user/eisenaj/code/UniversalUnconsciousness/_slurm/logs/delaseq_%j.err
##SBATCH --output=/home/eisenaj/code/UniversalUnconsciousness/_slurm/logs/delaseq_%j.log
##SBATCH --error=/home/eisenaj/code/UniversalUnconsciousness/_slurm/logs/delaseq_%j.err

cd /om2/user/eisenaj/code/UniversalUnconsciousness
source .venv/bin/activate

# python /home/eisenaj/code/UniversalUnconsciousness/UniversalUnconsciousness/DeLASE_analysis/delase_queueing_script.py
HYDRA_FULL_ERROR=1 python /om2/user/eisenaj/code/UniversalUnconsciousness/UniversalUnconsciousness/DeLASE_analysis/delase_queueing_script.py
