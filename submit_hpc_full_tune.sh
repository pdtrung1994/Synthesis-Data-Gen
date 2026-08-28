#!/bin/bash
#SBATCH --job-name=synth_full_tune
#SBATCH --output=logs/full_tune/slurm_%A_%a.out
#SBATCH --error=logs/full_tune/slurm_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=prod

# ==========================================
# AUTO-CREATE DIRECTORIES AND DYNAMIC SUBMISSION
# ==========================================
if [ -z "$SLURM_JOB_ID" ]; then
    mkdir -p logs/full_tune
    mkdir -p Results/full_tune
    
    export RESULTS_DIR="$PWD/Results/full_tune"
    
    echo "================================================="
    echo "Calculating total tasks for Full Tune..."
    CMD="singularity exec myenvironment.simg"
    
    # We must use the container to ensure python runs correctly if it's not installed locally
    TOTAL_TASKS=$(${CMD} python Runners/runner_full_tune.py --get_task_count | tail -n 1)
    if [ -z "$TOTAL_TASKS" ] || ! [[ "$TOTAL_TASKS" =~ ^[0-9]+$ ]]; then
        echo "Error: Could not calculate task count. Output was: $TOTAL_TASKS"
        exit 1
    fi
    MAX_INDEX=$((TOTAL_TASKS - 1))
    
    echo "Total Tasks Detected: $TOTAL_TASKS (Array: 0-$MAX_INDEX)"
    echo "================================================="
    echo "Submitting array job..."
    sbatch --export=ALL,RESULTS_DIR="$RESULTS_DIR" --array=0-$MAX_INDEX "$0" "$@"
    exit 0
fi
# ==========================================

echo "================================================="
echo "Starting Full Tune job on $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "================================================="

# Update the codebase before running
echo "Pulling latest changes from git..."
git pull origin main

# Use Singularity/Apptainer container instead of conda/virtualenv
CMD="singularity exec myenvironment.simg"
SCRIPT="Runners/runner_full_tune.py"

${CMD} python ${SCRIPT}

echo "================================================="
echo "Job completed."
echo "================================================="
