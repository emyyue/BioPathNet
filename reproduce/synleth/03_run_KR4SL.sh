#!/bin/bash
#SBATCH --job-name=run_KR4SL
#SBATCH --output=./slurm_out/run_KR4SL.out
#SBATCH --error=./slurm_out/run_KR4SL.err
#SBATCH --time=48:00:00
#SBATCH --mem=64Gb
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal

# To run this file:
# cd ./reproduce/synleth/
# sbatch ./scripts/02_run_KR4SL.sh


source miniconda3/etc/profile.d/conda.sh
conda activate synleth_env

export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.7/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.7/targets/x86_64-linux/lib


# Select seed and threshold
SEED=1234
THR=0.90
DIR_NAME="KR4SL_thr0$(echo "$THR" | sed 's/\.//; s/^0//')"
cd ${$DIR_NAME}/transductive


# Run KR4SL for the selected threshold and seed
python -W ignore train.py --suffix trans_reason --seed "$SEED"