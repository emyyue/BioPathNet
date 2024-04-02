#!/bin/bash
#SBATCH --job-name=nbfnet_biomed
#SBATCH --output=./slurm_out/pred_synleth_original_seeds_1-%j.txt
#SBATCH --error=./slurm_out/pred_synleth_original_seeds_1-%j.txt
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

ARRAY=("2023-12-07-10-15-45-671768:14"
	"2023-12-07-12-37-56-516186:10"
	"2023-12-07-12-40-13-719195:14"
	"2023-12-07-12-41-48-875654:15"
	"2023-12-07-12-44-21-539893:14")

for i in "${ARRAY[@]}"
do	
	python script/predict_multiple.py -c  config/knowledge_graph/synleth/synleth_trans_5layers_original_pred.yaml --gpus [0] --checkpoint /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/"${i%%:*}"/model_epoch_"${i##*:}".pth
done
