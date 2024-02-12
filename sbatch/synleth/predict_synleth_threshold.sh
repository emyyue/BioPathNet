#!/bin/bash
#SBATCH --job-name=nbfnet_biomed
#SBATCH --output=./slurm_out/pred_synleth_threshold_seeds-%j.txt
#SBATCH --error=./slurm_out/pred_synleth_threshold_seeds-%j.txt
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

ARRAY=("2024-02-05-20-14-31-970088:12"
"2024-02-05-20-31-23-541174:14"
"2024-02-05-20-48-22-787969:15"
"2024-02-05-21-05-39-433436:15"
"2024-02-05-21-23-01-525149:12")

for i in "${ARRAY[@]}"
do	
	python script/predict.py -c  config/knowledge_graph/synleth/synleth_trans_5layers_threshold_pred.yaml --gpus [0] --checkpoint /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/"${i%%:*}"/model_epoch_"${i##*:}".pth
done
