#!/bin/bash
#SBATCH --job-name=nbfnet_biomed
#SBATCH --output=./slurm_out/pred_cp.txt
#SBATCH --error=./slurm_out/pred_cp.txt
#SBATCH --time=8:00:00
#SBATCH --mem=64Gb
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
##SBATCH --constraint=a100_80gb
##SBATCH -w gpusrv61

split="cell_proliferation"

############################################################
model="2023-11-28-14-41-12-818389";epoch=9 # adv 1
layers=6
seed=42

CONDA_DIR=/home/icb/yue.hu/proj_genefun/conda-env/miniconda3
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

conda activate env_re_nbfnet

export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/targets/x86_64-linux/lib

python /home/icb/yue.hu/proj_genefun/NBFNet/script/predict.py \
    -c /home/icb/yue.hu/proj_genefun/NBFNet/config/knowledge_graph/primekg/eval/${split}_eval_${layers}_seed.yaml \
    --gpus [0] \
    --datasplit_seed $seed \
    --checkpoint  /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/$model/model_epoch_${epoch}.pth \
    --output_directory /home/icb/yue.hu/proj_genefun/NBFNet/experiments/txgnn_eval/$split/



