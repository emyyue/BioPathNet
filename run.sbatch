#!/bin/bash
#SBATCH --job-name=biopathnet-gfun
#SBATCH --output=slurm_out/job_out_%j.txt
#SBATCH --error=slurm_out/job_err_%j.txt
#SBATCH --time=12:00:00
#SBATCH --mem=32Gb
#SBATCH -c 4
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p

# Load Conda environment
CONDA_DIR=/home/icb/hui.cheng/miniconda3
source $CONDA_DIR/etc/profile.d/conda.sh
conda activate nbfnet

# Set the working directory
REPO_DIR=/lustre/groups/crna01/workspace/hui/BioPathNet
cd $REPO_DIR

# Run the script
python script/run.py -c config/mock/mockdata_run.yaml -s 42 --gpus [0] --version v1
