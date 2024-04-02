#!/bin/bash
#SBATCH --job-name=nbfnet_biomed
#SBATCH --output=./slurm_out/pred_cellProlif_%j.txt
#SBATCH --error=./slurm_out/pred_cellProlif_%j.txt
#SBATCH --time=2:00:00
#SBATCH --mem=64Gb
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
##SBATCH -w gpusrv61
#SBATCH --constraint=a100_80gb

CONDA_DIR=/home/icb/yue.hu/proj_genefun/conda-env/miniconda3
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

conda activate env_re_nbfnet

export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/targets/x86_64-linux/lib

cd /home/icb/yue.hu/proj_genefun/NBFNet

#before 2024 - Jan
#python script/predict.py -c  config/knowledge_graph/primekg/vis/cell_proliferation_vis.yaml --gpus [0] --checkpoint /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/2023-10-11-18-20-38-624703/model_epoch_10.pth
# seed 42
#python script/predict.py -c  config/knowledge_graph/primekg/vis/cell_proliferation_pred.yaml --gpus [0] --checkpoint /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/2023-11-28-14-41-12-818389/model_epoch_9.pth

# 14
python script/predict.py -c  config/knowledge_graph/primekg/vis/cell_proliferation_pred.yaml --gpus [0] --checkpoint /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/2023-11-28-14-40-46-486490/model_epoch_9.pth


# seed 1618
python script/predict.py -c  config/knowledge_graph/primekg/vis/cell_proliferation_pred.yaml --gpus [0] --checkpoint /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/2023-11-28-14-40-12-551111/model_epoch_9.pth

#seed 2011
python script/predict.py -c  config/knowledge_graph/primekg/vis/cell_proliferation_pred.yaml --gpus [0] --checkpoint /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/2023-11-28-14-40-42-549856/model_epoch_10.pth

# seed 88
python script/predict.py -c  config/knowledge_graph/primekg/vis/cell_proliferation_pred.yaml --gpus [0] --checkpoint /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/2023-11-28-14-37-54-363599/model_epoch_9.pth
#seed 314
python script/predict.py -c  config/knowledge_graph/primekg/vis/cell_proliferation_pred.yaml --gpus [0] --checkpoint /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/2023-12-02-09-30-22-638385/model_epoch_9.pth

