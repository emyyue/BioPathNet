#!/bin/bash


for i in {1..7}
do
    sbatch --output=./slurm_out/job_output${i}.txt --error=./slurm_out/job_error${i}.txt run_biomed.sh ../config/knowledge_graph/hyperparam/biomed_joint_${i}.yaml
done
