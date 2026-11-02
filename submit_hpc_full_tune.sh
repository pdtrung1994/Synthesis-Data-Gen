#!/bin/bash
#SBATCH --job-name=synth_full_tune
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=144:00:00
#SBATCH --partition=prod

# ==========================================
# AUTO-CREATE DIRECTORIES AND DYNAMIC SUBMISSION
# ==========================================
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    if [ -n "$SLURM_JOB_ID" ]; then
        echo "[!] Detected non-array SLURM allocation (Job $SLURM_JOB_ID)."
        echo "[!] Tip: run 'bash submit_hpc_full_tune.sh' directly on the login node. Attempting array submission now..."
    fi
    mkdir -p logs
    mkdir -p Results
    RUN_NUM=1
    while [ -d "logs/full_tune_$(printf "%02d" $RUN_NUM)" ] || [ -d "Results/full_tune_$(printf "%02d" $RUN_NUM)" ]; do
        RUN_NUM=$((RUN_NUM + 1))
    done
    RUN_DIR_NAME="full_tune_$(printf "%02d" $RUN_NUM)"
    
    mkdir -p "logs/$RUN_DIR_NAME"
    mkdir -p "Results/$RUN_DIR_NAME"
    
    export RESULTS_DIR="$PWD/Results/$RUN_DIR_NAME"
    export LOG_DIR="$PWD/logs/$RUN_DIR_NAME"
    
    echo "================================================="
    echo "Results directory: $RESULTS_DIR"
    echo "Logs directory:    $LOG_DIR"
    echo "Calculating total tasks for Full Tune..."
    CMD="singularity exec myenvironment.simg"
    
    # We must use the container to ensure python runs correctly if it's not installed locally
    TOTAL_TASKS=$(${CMD} python3 Runners/runner_full_tune.py --get_task_count | tail -n 1)
    if [ -z "$TOTAL_TASKS" ] || ! [[ "$TOTAL_TASKS" =~ ^[0-9]+$ ]]; then
        echo "Error: Could not calculate task count. Output was: $TOTAL_TASKS"
        exit 1
    fi
    MAX_INDEX=$((TOTAL_TASKS - 1))
    
    echo "Total Tasks Detected: $TOTAL_TASKS"
    echo "================================================="
    echo "Submitting array jobs in chunks of 900 to avoid MaxArraySize limits..."
    
    CHUNK_SIZE=900
    for (( i=0; i<$TOTAL_TASKS; i+=$CHUNK_SIZE )); do
        END_IDX=$((i + CHUNK_SIZE - 1))
        if [ $END_IDX -gt $MAX_INDEX ]; then
            END_IDX=$MAX_INDEX
        fi
        
        ARRAY_START=0
        ARRAY_END=$((END_IDX - i))
        
        echo "Submitting chunk offset $i (Array: 0-$ARRAY_END)"
        if ! sbatch --export=ALL,RESULTS_DIR="$RESULTS_DIR",LOG_DIR="$LOG_DIR",TASK_OFFSET="$i" --output="logs/$RUN_DIR_NAME/slurm_%A_%a.out" --error="logs/$RUN_DIR_NAME/slurm_%A_%a.err" --array=0-$ARRAY_END "$0" "$@"; then
            echo "Error: sbatch submission failed. Please run 'bash submit_hpc_full_tune.sh' from the login node."
            exit 1
        fi
    done
    exit 0
fi
# ==========================================

echo "================================================="
echo "Starting Full Tune job on $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "================================================="

# Use Singularity/Apptainer container instead of conda/virtualenv
CMD="singularity exec myenvironment.simg"
SCRIPT="Runners/runner_full_tune.py"

${CMD} python3 ${SCRIPT}

echo "================================================="
echo "Job completed."
echo "================================================="
