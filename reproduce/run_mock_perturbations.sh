#!/bin/bash
#SBATCH --job-name=biopathnet_biomed_mock_perturbations
#SBATCH --output=/lustre/groups/crna01/projects/synthetic_lethality/BioPathNet/slurm_out/run_mock_perturbations.txt
#SBATCH --error=/lustre/groups/crna01/projects/synthetic_lethality/BioPathNet/slurm_out/run_mock_perturbations.err
#SBATCH --time=01:00:00
#SBATCH --mem=120Gb
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --constraint=a100_40gb


CONDA_DIR=/home/icb/svitlana.oleshko/miniconda3
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate biopathnet

cd /lustre/groups/crna01/projects/synthetic_lethality/BioPathNet

# Log the start time
start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script started at: $start_time"

DATA_PATH=/lustre/groups/crna01/projects/synthetic_lethality/BioPathNet/data/mock
SEED=123

ARRAY=(
       "train1:remove_top_kth_relation:1"
       "train1:remove_top_kth_relation:2"
       "train1:remove_top_kth_relation:3"
       "train1:remove_top_kth_relation:4"
       "train1:remove_top_kth_relation:5"
       "train1:remove_random_relations:10"
       "train1:remove_random_relations:20"
       "train1:remove_random_relations:50"
       "train1:remove_random_relations:80"
       "train1:add_random_relations:10"
       "train1:add_random_relations:20"
       "train1:add_random_relations:50"
       "train1:add_random_relations:80"
       "train1:remove_top_nodes:5"

       "train2:remove_top_nodes:1"
       "train2:remove_top_nodes:3"
       "train2:remove_top_nodes:5"
    
       ## Optional
       # "train2:remove_random_relations:10"
       # "train2:remove_random_relations:20"
       # "train2:remove_random_relations:50"
       # "train2:remove_random_relations:80"
       # "train2:add_random_relations:10"
       # "train2:add_random_relations:20"
       # "train2:add_random_relations:50"
       # "train2:add_random_relations:80"
    )

for i in "${ARRAY[@]}"
do
  IFS=':' read -r which_graph perturbation_mode k <<< "$i"
  
  # Run the perturb_data.py script
  python script/perturb_data.py --seed "$SEED" --data_path "$DATA_PATH" --which_graph "$which_graph" --perturbation_mode "$perturbation_mode" --k "$k"
  
  # Copy the mockdata_run.yaml file and rename it
  new_yaml="config/mock/mockdata_run_${which_graph}_${perturbation_mode}_${k}.yaml"
  cp config/mock/mockdata_run.yaml "$new_yaml"
  
  # Modify the new config YAML file based on which_graph
  if [[ "$which_graph" == "train1" ]]; then
    sed -i "s/files: \['train1.txt', 'train2.txt', 'valid.txt', 'test.txt'\]/files: \['train1_${perturbation_mode}_${k}.txt', 'train2.txt', 'valid.txt', 'test.txt'\]/" "$new_yaml"
  elif [[ "$which_graph" == "train2" ]]; then
    sed -i "s/files: \['train1.txt', 'train2.txt', 'valid.txt', 'test.txt'\]/files: \['train1.txt', 'train2_${perturbation_mode}_${k}.txt', 'valid.txt', 'test.txt'\]/" "$new_yaml"
  fi
  
  # Run the run.py script with the new config YAML file
  python script/run.py -s 1234 -c "$new_yaml" --gpus [0]
done
  
# Log the end time
end_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script ended at: $end_time"
