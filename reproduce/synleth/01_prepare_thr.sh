#!/bin/bash
#SBATCH --job-name=prepare_thr
#SBATCH --output=./slurm_out/prepare_thr.out
#SBATCH --error=./slurm_out/prepare_thr.err
#SBATCH --time=00:10:00
#SBATCH --mem=16Gb
#SBATCH -c 4
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal

# To run this file:
# cd ./reproduce/synleth/
# sbatch ./scripts/01_prepare_thr.sh


# Select threshold
THR=0.90

DIR_NAME="KR4SL_thr0$(echo "$THR" | sed 's/\.//; s/^0//')"

# Create the new directory for the selected threshold
mkdir -p "$DIR_NAME"

DIRS_TO_COPY=("data" "results" "transductive")

# Copy each required directory into the new directory
for dir in "${DIRS_TO_COPY[@]}"; do
    if [ -d "KR4SL/$dir" ]; then
        cp -r "KR4SL/$dir" "$DIR_NAME"
    else
        echo "Directory $dir does not exist, skipping copy."
    fi
done