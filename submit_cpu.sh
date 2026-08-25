#!/bin/bash
#SBATCH --job-name=synth_data_cpu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=12:00:00
#SBATCH --partition=prod

# Create logs directory if it doesn't exist
mkdir -p logs

echo "================================================="
echo "Starting job on $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "================================================="

# Update the codebase before running
echo "Pulling latest changes from git..."
git pull origin main

# Use Singularity/Apptainer container instead of conda/virtualenv
CMD="singularity exec myenvironment.simg"
SCRIPT="Runners/runner.py"

${CMD} python3 ${SCRIPT}

echo "================================================="
echo "Job completed."
echo "================================================="
