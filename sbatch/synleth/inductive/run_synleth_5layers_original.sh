#!/bin/bash
#SBATCH --job-name=nbfnet_biomed
#SBATCH --output=./slurm_out/run_synleth_seeds-%j.txt
#SBATCH --error=./slurm_out/run_synleth_seeds-%j.txt
#SBATCH --time=48:00:00
#SBATCH --mem=64Gb
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --constraint=a100_80gb

CONDA_DIR=/home/icb/yue.hu/proj_genefun/conda-env/miniconda3
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

conda activate env_re_nbfnet

export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/targets/x86_64-linux/lib

cd /home/icb/yue.hu/proj_genefun/NBFNet


for i in 1234 1235 1236 1237 1238
do
	echo $i
	python script/run.py -s $i -c  config/knowledge_graph/synleth/inductive/synleth_inductive_5layers_original.yaml --gpus [0]
done
