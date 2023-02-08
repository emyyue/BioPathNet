#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=./slurm_out/job_output.txt
#SBATCH --error=./slurm_out/job_error.txt
#SBATCH --time=24:00:00 # takes around 7 hours to run
#SBATCH --mem=24Gb
#SBATCH -c 4
#SBATCH -w supergpu07
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu


#module load python
#source ~/allvirtualenvs/nbfnet/bin/activate # activate the virtualenv
#module load cuda/11.1
#cd /home/mila/y/yue.hu/github/NBFNet


cd /lustre/groups/crna01/projects/genefunction/
BASE=/lustre/groups/crna01/projects/genefunction/
CONDA_DIR=$BASE/miniconda3
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate env_nbfnet
cd NBFNet

export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.7/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.7/targets/x86_64-linux/lib

#python -m torch.distributed.launch --nproc_per_node=4 script/run.py -c config/knowledge_graph/biogrid.yaml --gpus [0,1]

python script/run.py -c config/knowledge_graph/submission_PC_factgraph.yaml --gpus [0] --version v1

