#!/bin/bash
#SBATCH --job-name=nbfnet_biomed
#SBATCH --output=./slurm_out/job_output.txt
#SBATCH --error=./slurm_out/job_error.txt
#SBATCH --time=12:00:00
#SBATCH --mem=24Gb
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --partition=main

# TODO: change to how to activate environment
module load python
source ~/allvirtualenvs/nbfnet/bin/activate 
module load cuda/11.1
cd /home/mila/y/yue.hu/github/NBFNet


python script/run.py -c ${1} --gpus [0] --version v1