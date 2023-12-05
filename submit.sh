#!/bin/bash
#
#SBATCH --job-name=VAE
#SBATCH --account=cosmo_ai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
source /home/dongx/.bashrc
conda activate /home/dongx/anaconda3/envs/env_pytorch
# cd /home/dongx/UNET-64/ML-Recon
export PYTHONPATH="$PWD/Unet"
export PYTHONPATH="/home/dongx/anaconda3/envs/env_pytorch/lib/python3.6/site-packages"
#srun python3 /home/dongx/UNET-64/ML-Recon/plot_custom_2_output.py
srun python3 vae-grid.py