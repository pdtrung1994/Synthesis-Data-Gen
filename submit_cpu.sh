#!/bin/bash
#SBATCH --job-name=synth_data_cpu
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=48:00:00
#SBATCH --partition=prod
#SBATCH --array=0-8

# ==========================================
# TỰ ĐỘNG TẠO THƯ MỤC LOG VÀ SUBMIT
# ==========================================
if [ -z "$SLURM_JOB_ID" ]; then
    mkdir -p logs
    mkdir -p Results
    RUN_NUM=1
    while [ -d "logs/run_$(printf "%02d" $RUN_NUM)" ] || [ -d "Results/run_$(printf "%02d" $RUN_NUM)" ]; do
        RUN_NUM=$((RUN_NUM + 1))
    done
    RUN_DIR_NAME="run_$(printf "%02d" $RUN_NUM)"
    
    mkdir -p "logs/$RUN_DIR_NAME"
    mkdir -p "Results/$RUN_DIR_NAME"
    
    export RESULTS_DIR="$PWD/Results/$RUN_DIR_NAME"
    
    echo "================================================="
    echo "Tạo thư mục lưu log: logs/$RUN_DIR_NAME"
    echo "Tạo thư mục lưu results: Results/$RUN_DIR_NAME"
    echo "Đang submit job..."
    echo "================================================="
    sbatch --export=ALL,RESULTS_DIR="$RESULTS_DIR" --output="logs/$RUN_DIR_NAME/slurm_%A_%a.out" --error="logs/$RUN_DIR_NAME/slurm_%A_%a.err" "$0" "$@"
    exit 0
fi
# ==========================================

# Mảng chứa danh sách 5 databases
DATASETS=("Plant_oil" "Brewed_vinegar" "Wine_spoilage" "Chinese_wine" "Coffee")

# Lấy ra database tương ứng với Task ID của Array Job
# Nếu chạy không qua sbatch (không có SLURM_ARRAY_TASK_ID), mặc định lấy index 0
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [ $TASK_ID -le 4 ]; then
    CURRENT_DATASET=${DATASETS[$TASK_ID]}
    if [ "$CURRENT_DATASET" == "Coffee" ]; then
        # Coffee khá nhỏ nên chạy gộp tất cả các seed vào chung 1 job (Task 4)
        SEEDS="1 2 5 10 20"
    else
        SEEDS="1 2 5 10"
    fi
else
    # Task 5 đến 8 tương ứng với Dataset 0 đến 3 (chỉ chạy seed 20)
    DATASET_INDEX=$((TASK_ID - 5))
    CURRENT_DATASET=${DATASETS[$DATASET_INDEX]}
    SEEDS="20"
fi

echo "================================================="
echo "Starting job on $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Processing dataset: $CURRENT_DATASET"
echo "Processing seeds: $SEEDS"
echo "================================================="

# Update the codebase before running
echo "Pulling latest changes from git..."
git pull origin main

# Use Singularity/Apptainer container instead of conda/virtualenv
CMD="singularity exec myenvironment.simg"
SCRIPT="Runners/runner.py"

${CMD} python3 ${SCRIPT} --datasets "$CURRENT_DATASET" --seeds $SEEDS

echo "================================================="
echo "Job completed."
echo "================================================="
