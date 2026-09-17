#!/bin/bash
#SBATCH --job-name=CNT_train
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./out.log

set -euo pipefail

# Store start time
START_TIME=$SECONDS

echo "---------------------------------------------"
echo "                START: $(date)"
echo "---------------------------------------------"

python -m cnt_project.model_development.training.runners.hparam_search_runner

# Store end time
END_TIME=$SECONDS

echo "---------------------------------------------"
echo "                END: $(date)"
echo "---------------------------------------------"

# Calculate elapsed time
ELAPSED_TIME=$((END_TIME - START_TIME))

# Convert to human-readable format (hours, minutes, seconds)
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))

echo "---------------------------------------------"
echo "    TOTAL EXECUTION TIME: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "---------------------------------------------"
