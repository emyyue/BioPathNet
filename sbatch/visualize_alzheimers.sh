#!/bin/bash
#SBATCH --job-name=nbfnet_biomed
#SBATCH --output=./slurm_out/vis_alz.txt
#SBATCH --error=./slurm_out/vis_alz.txt
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

python script/visualize_graph.py -c  config/knowledge_graph/primekg/alzheimers_vis.yaml --gpus [0] --checkpoint /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet//2023-11-13-17-50-17-020588/model_epoch_3.pth

#python script/predict.py -c  config/knowledge_graph/primekg/alzheimers_vis.yaml --gpus [0] --checkpoint /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet//2023-11-13-17-50-17-020588/model_epoch_3.pth
