#!/bin/bash
#SBATCH --job-name=nbfnet_biomed
#SBATCH --output=./slurm_out/eval_mh.txt
#SBATCH --error=./slurm_out/eval_mh.txt
#SBATCH --time=2:00:00
#SBATCH --mem=64Gb
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
##SBATCH -w gpusrv61

CONDA_DIR=/home/icb/yue.hu/proj_genefun/conda-env/miniconda3
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

conda activate env_re_nbfnet

export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/targets/x86_64-linux/lib


#python ../script/txgnn_evaluate.py -c /home/icb/yue.hu/proj_genefun/NBFNet/config/knowledge_graph/primekg/mental_health_eval.yaml --gpus [0] --checkpoint  /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/2023-09-28-16-05-20-867235/model_epoch_8.pth --output_directory /home/icb/yue.hu/proj_genefun/NBFNet/experiments/txgnn_eval/mental_health/

### TxGNN part
source /home/icb/samuele.firmani/miniconda3/bin/activate txgnn_env

cd /home/icb/yue.hu/proj_genefun/source/txgnn_nbfnet/scripts

python txgnn_nbfnet_evaluation.py

python  get_txgnn_evaluation.py
