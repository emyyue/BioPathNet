#!/bin/bash
#SBATCH --job-name=nbfnet_biomed
#SBATCH --output=./slurm_out/eval_cardio_seeds.txt
#SBATCH --error=./slurm_out/eval_cardio_seeds.txt
#SBATCH --time=8:00:00
#SBATCH --mem=64Gb
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
##SBATCH --constraint=a100_80gb
##SBATCH -w gpusrv61

split="cardiovascular"

############################################################
model="2023-11-30-09-03-27-134157"
epoch=10
layers=6
seed=14

CONDA_DIR=/home/icb/yue.hu/proj_genefun/conda-env/miniconda3
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

conda activate env_re_nbfnet

export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/targets/x86_64-linux/lib

mkdir -p /home/icb/yue.hu/proj_genefun/NBFNet/experiments/txgnn_eval/$split/
python /home/icb/yue.hu/proj_genefun/NBFNet/script/txgnn_evaluate.py \
    -c /home/icb/yue.hu/proj_genefun/NBFNet/config/knowledge_graph/primekg/eval/${split}_eval_${layers}_seed.yaml \
    --gpus [0] \
    --datasplit_seed $seed \
    --checkpoint  /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/$model/model_epoch_${epoch}.pth \
    --output_directory /home/icb/yue.hu/proj_genefun/NBFNet/experiments/txgnn_eval/$split/

### TxGNN part
conda deactivate
conda activate txgnn_env_plotnine

python /home/icb/yue.hu/proj_genefun/source/txgnn_nbfnet/scripts/txgnn_nbfnet_evaluation.py  \
    $split \
    /home/icb/samuele.firmani/NBFNet/sbatch/primekg/$split/txgnn_logs/saved_models/${split}_model_ckpt_best_hyperparam/ \
    /home/icb/yue.hu/proj_genefun/NBFNet/experiments/txgnn_eval/$split/$model \
    ${split}_plot_${layers}layers.pdf




#############################################################
model="2023-11-30-09-09-13-617387"
epoch=9 
layers=6
seed=42

CONDA_DIR=/home/icb/yue.hu/proj_genefun/conda-env/miniconda3
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

conda activate env_re_nbfnet

export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/targets/x86_64-linux/lib

mkdir -p /home/icb/yue.hu/proj_genefun/NBFNet/experiments/txgnn_eval/$split/
python /home/icb/yue.hu/proj_genefun/NBFNet/script/txgnn_evaluate.py \
    -c /home/icb/yue.hu/proj_genefun/NBFNet/config/knowledge_graph/primekg/eval/${split}_eval_${layers}_seed.yaml \
    --gpus [0] \
    --datasplit_seed $seed \
    --checkpoint  /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/$model/model_epoch_${epoch}.pth \
    --output_directory /home/icb/yue.hu/proj_genefun/NBFNet/experiments/txgnn_eval/$split/

### TxGNN part
conda deactivate
conda activate txgnn_env_plotnine

python /home/icb/yue.hu/proj_genefun/source/txgnn_nbfnet/scripts/txgnn_nbfnet_evaluation.py  \
    $split \
    /home/icb/samuele.firmani/NBFNet/sbatch/primekg/$split/txgnn_logs/saved_models/${split}_model_ckpt_best_hyperparam_${seed}/ \
    /home/icb/yue.hu/proj_genefun/NBFNet/experiments/txgnn_eval/$split/$model \
    ${split}_plot_${layers}layers.pdf


#############################################################
model="2023-11-30-09-14-50-521193"
epoch=9 
layers=6
seed=88


conda deactivate
conda activate env_re_nbfnet

export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/targets/x86_64-linux/lib

mkdir -p /home/icb/yue.hu/proj_genefun/NBFNet/experiments/txgnn_eval/$split/
python /home/icb/yue.hu/proj_genefun/NBFNet/script/txgnn_evaluate.py \
    -c /home/icb/yue.hu/proj_genefun/NBFNet/config/knowledge_graph/primekg/eval/${split}_eval_${layers}_seed.yaml \
    --gpus [0] \
    --datasplit_seed ${seed} \
    --checkpoint  /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/$model/model_epoch_${epoch}.pth \
    --output_directory /home/icb/yue.hu/proj_genefun/NBFNet/experiments/txgnn_eval/$split/

### TxGNN part
conda deactivate
conda activate txgnn_env_plotnine

python /home/icb/yue.hu/proj_genefun/source/txgnn_nbfnet/scripts/txgnn_nbfnet_evaluation.py  \
    $split \
    /home/icb/samuele.firmani/NBFNet/sbatch/primekg/$split/txgnn_logs/saved_models/${split}_model_ckpt_best_hyperparam_${seed}/ \
    /home/icb/yue.hu/proj_genefun/NBFNet/experiments/txgnn_eval/$split/$model \
    ${split}_plot_${layers}layers.pdf


#############################################################
model="2023-11-30-09-11-58-639129"
epoch=5
layers=6
seed=1618

conda deactivate
conda activate env_re_nbfnet

export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/targets/x86_64-linux/lib

mkdir -p /home/icb/yue.hu/proj_genefun/NBFNet/experiments/txgnn_eval/$split/
python /home/icb/yue.hu/proj_genefun/NBFNet/script/txgnn_evaluate.py \
    -c /home/icb/yue.hu/proj_genefun/NBFNet/config/knowledge_graph/primekg/eval/${split}_eval_${layers}_seed.yaml \
    --gpus [0] \
    --datasplit_seed ${seed} \
    --checkpoint  /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/$model/model_epoch_${epoch}.pth \
    --output_directory /home/icb/yue.hu/proj_genefun/NBFNet/experiments/txgnn_eval/$split/

### TxGNN part
conda deactivate
conda activate txgnn_env_plotnine

python /home/icb/yue.hu/proj_genefun/source/txgnn_nbfnet/scripts/txgnn_nbfnet_evaluation.py  \
    $split \
    /home/icb/samuele.firmani/NBFNet/sbatch/primekg/$split/txgnn_logs/saved_models/${split}_model_ckpt_best_hyperparam_${seed}/ \
    /home/icb/yue.hu/proj_genefun/NBFNet/experiments/txgnn_eval/$split/$model \
    ${split}_plot_${layers}layers.pdf

#############################################################
model="2023-11-30-10-27-38-244662"
epoch=9
layers=6
seed=2011

CONDA_DIR=/home/icb/yue.hu/proj_genefun/conda-env/miniconda3
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

conda activate env_re_nbfnet

export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/targets/x86_64-linux/lib

mkdir -p /home/icb/yue.hu/proj_genefun/NBFNet/experiments/txgnn_eval/$split/
python /home/icb/yue.hu/proj_genefun/NBFNet/script/txgnn_evaluate.py \
    -c /home/icb/yue.hu/proj_genefun/NBFNet/config/knowledge_graph/primekg/eval/${split}_eval_${layers}_seed.yaml \
    --gpus [0] \
    --datasplit_seed $seed \
    --checkpoint  /home/icb/yue.hu/proj_genefun/NBFNet/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/$model/model_epoch_${epoch}.pth \
    --output_directory /home/icb/yue.hu/proj_genefun/NBFNet/experiments/txgnn_eval/$split/

### TxGNN part
conda deactivate
conda activate txgnn_env_plotnine

python /home/icb/yue.hu/proj_genefun/source/txgnn_nbfnet/scripts/txgnn_nbfnet_evaluation.py  \
    $split \
    /home/icb/samuele.firmani/NBFNet/sbatch/primekg/$split/txgnn_logs/saved_models/${split}_model_ckpt_best_hyperparam_${seed}/ \
    /home/icb/yue.hu/proj_genefun/NBFNet/experiments/txgnn_eval/$split/$model \
    ${split}_plot_${layers}layers.pdf

