#!/bin/bash
#SBATCH --job-name=synth_data_cpu
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=12:00:00
#SBATCH --partition=prod
#SBATCH --array=0-4

# Mảng chứa danh sách 5 databases
DATASETS=("Plant_oil" "Brewed_vinegar" "Wine_spoilage" "Chinese_wine" "Coffee")

# Lấy ra database tương ứng với Task ID của Array Job
# Nếu chạy không qua sbatch (không có SLURM_ARRAY_TASK_ID), mặc định lấy index 0
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
CURRENT_DATASET=${DATASETS[$TASK_ID]}

echo "================================================="
echo "Starting job on $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Processing dataset: $CURRENT_DATASET"
echo "================================================="

# Update the codebase before running
echo "Pulling latest changes from git..."
git pull origin main

# Use Singularity/Apptainer container instead of conda/virtualenv
CMD="singularity exec myenvironment.simg"
SCRIPT="Runners/runner.py"

${CMD} python3 ${SCRIPT} --datasets "$CURRENT_DATASET"

echo "================================================="
echo "Job completed."
echo "================================================="
