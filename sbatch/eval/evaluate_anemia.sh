#!/bin/bash
#SBATCH --job-name=nbfnet_biomed
#SBATCH --output=./slurm_out/eval_cp.txt
#SBATCH --error=./slurm_out/eval_cp.txt
#SBATCH --time=2:00:00
#SBATCH --mem=64Gb
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


python /home/icb/yue.hu/proj_genefun/NBFNet/script/txgnn_evaluate.py -c /home/icb/yue.hu/proj_genefun/NBFNet/config/knowledge_graph/primekg/eval/cell_proliferation_eval.yaml --gpus [0] --checkpoint  /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/2023-10-11-18-20-38-815410/model_epoch_10.pth --output_directory /home/icb/yue.hu/proj_genefun/NBFNet/experiments/txgnn_eval/cell_proliferation/

### TxGNN part
conda deactivate
conda activate txgnn_env_plotnine

python /home/icb/yue.hu/proj_genefun/source/txgnn_nbfnet/scripts/txgnn_nbfnet_evaluation.py  cell_proliferation /home/icb/samuele.firmani/NBFNet/sbatch/primekg/cell_proliferation/txgnn_logs/saved_models/cell_proliferation_model_ckpt_best_hyperparam/ /home/icb/yue.hu/proj_genefun/NBFNet/experiments/txgnn_eval/cell_proliferation/ plot_6layers.pdf

#python /home/icb/yue.hu/proj_genefun/source/txgnn_nbfnet/scripts/txgnn_nbfnet_evaluation.py   /home/icb/yue.hu/proj_genefun/NBFNet/experiments/txgnn_eval/cell_proliferation/plot_6layers.pdf

