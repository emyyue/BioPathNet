#!/bin/bash
#SBATCH --job-name=nbfnet_biomed
#SBATCH --output=./slurm_out/pred_cp_woLeak-%j.txt
#SBATCH --error=./slurm_out/pred_cp_woLeak-%j.txt
#SBATCH --time=2:00:00
#SBATCH --mem=64Gb
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
##SBATCH -w gpusrv61
#SBATCH --constraint=a100_80gb


time=`date +%Y%m%d-%H%M%S`

CONDA_DIR=/home/icb/yue.hu/proj_genefun/conda-env/miniconda3
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

conda activate env_re_nbfnet

export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/targets/x86_64-linux/lib

cd /home/icb/yue.hu/proj_genefun/NBFNet

#before 2024 - Jan
#python script/predict.py -c  config/knowledge_graph/primekg/vis/cell_proliferation_vis.yaml --gpus [0] --checkpoint /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/2023-10-11-18-20-38-624703/model_epoch_10.pth

python script/predict.py -c  config/knowledge_graph/primekg/vis/cell_proliferation_pred_woLeak.yaml --gpus [0] --checkpoint /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/2023-11-28-14-41-12-818389/model_epoch_9.pth


#python script/visualize.py -c  config/knowledge_graph/primekg/vis/cell_proliferation_vis.yaml --gpus [0] --checkpoint /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/2023-10-11-18-20-38-624703//model_epoch_10.pth

