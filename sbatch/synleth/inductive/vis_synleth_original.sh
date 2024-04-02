#!/bin/bash
#SBATCH --job-name=nbfnet_biomed
#SBATCH --output=./slurm_out/vis_synleth_original_seeds-%j.txt
#SBATCH --error=./slurm_out/vis_synleth_original_seeds-%j.txt
#SBATCH --time=4:00:00
#SBATCH --mem=128Gb
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --constraint=a100_80gb
##SBATCH -w gpusrv61

CONDA_DIR=/home/icb/yue.hu/proj_genefun/conda-env/miniconda3
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

conda activate env_re_nbfnet

export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/targets/x86_64-linux/lib

cd /home/icb/yue.hu/proj_genefun/NBFNet

ARRAY=("2024-02-09-09-57-55-848775:14")

for i in "${ARRAY[@]}"
do	
	python script/visualize.py -c  config/knowledge_graph/synleth/inductive/synleth_inductive_5layers_original_pred.yaml --gpus [0] --checkpoint /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/"${i%%:*}"/model_epoch_"${i##*:}".pth
	python script/visualize_graph.py -c config/knowledge_graph/synleth/inductive/synleth_inductive_5layers_original_pred.yaml --gpus [0] --checkpoint /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/"${i%%:*}"/model_epoch_"${i##*:}".pth
done
