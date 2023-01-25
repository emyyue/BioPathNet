#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --output=./slurm_out/job_output_pred.txt
#SBATCH --error=./slurm_out/job_error_pred.txt
#SBATCH --time=02:00:00
#SBATCH --mem=24Gb
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --partition=main


#module load python
#source ~/allvirtualenvs/nbfnet/bin/activate # activate the virtualenv
#module load cuda/11.1
#cd /home/mila/y/yue.hu/github/NBFNet


cd /lustre/groups/crna01/projects/genefunction/
BASE=/lustre/groups/crna01/projects/genefunction/
CONDA_DIR=$BASE/miniconda3
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate env_nbfnet
cd NBFNet

python script/predict.py -c config/knowledge_graph/qki_factgraph.yaml --gpus [0] --version v1 --checkpoint /lustre/groups/crna01/projects/genefunction/experiments/KnowledgeGraphCompletionBiomed/biomedical/NBFNet/qki_train_epoch8/model_epoch_6.pth
#python script/predict.py -c config/knowledge_graph/datachall.yaml --gpus [0] --version v1 --checkpoint /home/icb/yue.hu/proj_genefun/experiments/KnowledgeGraphCompletionBiomed/biomedical/TransE/2023-01-24-16-45-07/model_epoch_1.pth