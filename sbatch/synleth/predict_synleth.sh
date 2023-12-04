#!/bin/bash
#SBATCH --job-name=nbfnet_biomed
#SBATCH --output=./slurm_out/pred_synleth.txt
#SBATCH --error=./slurm_out/pred_synleth.txt
#SBATCH --time=2:00:00
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

for i in 2023-12-03-14-46-06-706375 2023-12-03-15-11-19-848826 2023-12-03-15-35-36-308520 2023-12-03-15-58-08-481286 2023-12-03-16-20-38-749801
do	
	python script/predict.py -c  config/knowledge_graph/synleth/synleth_trans_5layers_threshold_eval.yaml --gpus [0] --checkpoint /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/$i/model_epoch_18.pth
done
