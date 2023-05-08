#!/bin/bash


for i in {1..7}
do
    sbatch script_run_biomed.sh config/knowledge_graph/hyperparam/biomed_joint_${i}.yaml
done