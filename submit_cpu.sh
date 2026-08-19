#!/bin/bash
#SBATCH --job-name=synth_data_cpu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --time=24:00:00
#SBATCH --partition=prod

# Create logs directory if it doesn't exist
mkdir -p logs

echo "================================================="
echo "Starting job on $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "================================================="

# Activate Conda environment
source ~/.bashrc
conda activate myenv

# Run the python orchestrator
# The runner.py will automatically detect SLURM_CPUS_PER_TASK and optimize itself
# test github

python Runners/runner.py

echo "================================================="
echo "Job completed."
echo "================================================="
