import os
import subprocess
import concurrent.futures
import sys
import argparse
from tqdm import tqdm
import multiprocessing

# Configure paths
EXPERIMENT_SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'experiment.py')
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')

# ==============================================================================
# ENVIRONMENT DETECTION (Local vs HPC)
# ==============================================================================
IS_HPC = "SLURM_JOB_ID" in os.environ

if IS_HPC:
    # CRITICAL HPC CPU TUNING: Prevent CPU oversubscription
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["N_JOBS"] = "1" 
else:
    # Original behavior for local execution: Use all cores for HP-Tuning
    os.environ["N_JOBS"] = "-1"

def run_task(task_kwargs):
    dataset = task_kwargs['dataset']
    seed = task_kwargs['seed']
    pbar = task_kwargs['pbar']
    n_splits = task_kwargs.get('n_splits', 10)
    
    target_fold = task_kwargs.get('fold', -1)
    
    log_file_name = f"{dataset}_S{seed}.log"
    if target_fold != -1:
        log_file_name = f"{dataset}_S{seed}_F{target_fold}.log"
        
    log_file = os.path.join(LOG_DIR, log_file_name)
    
    # Open log file to write output directly
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Starting {dataset} - Seed {seed} - Fold {target_fold if target_fold != -1 else 'All'}\n\n")
        f.flush()
        
        folds_to_run = range(1, n_splits + 1) if target_fold == -1 else [target_fold]
        
        for fold in folds_to_run:
            command = [sys.executable, EXPERIMENT_SCRIPT, "--dataset", dataset, "--seed", str(seed), "--fold", str(fold)]
            process = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, text=True)
            process.wait()
            
            # If running sequentially across folds in a single task, update pbar here
            if target_fold == -1:
                pbar.update(1)
            
            if process.returncode != 0:
                return f"[!] ERROR in {dataset} (Seed {seed}, Fold {fold}) - See details at {log_file}"
                
        # If running a single fold task, update pbar after completion
        if target_fold != -1:
            pbar.update(1)
                
    return f"[+] COMPLETED {dataset} (Seed {seed}{f', Fold {target_fold}' if target_fold != -1 else ''})"

if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    
    DEFAULT_DATASETS = ["Plant_oil", "Brewed_vinegar", "Wine_spoilage", "Chinese_wine", "Coffee"]
    DEFAULT_SEEDS = [1, 2, 5, 10, 20]
    
    # In HPC, SLURM_CPUS_PER_TASK is the number of cores requested.
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    default_workers = int(slurm_cpus) if slurm_cpus else 6 # Original default was 6
    
    parser = argparse.ArgumentParser(description="Multi-process Orchestrator for Experiments")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, 
                        help="List of datasets to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="List of seeds to run")
    parser.add_argument("--max_workers", type=int, default=default_workers,
                        help=f"Maximum number of parallel workers (default: {default_workers})")
                        
    args = parser.parse_args()
    
    n_splits = 10
    total_folds = len(args.datasets) * len(args.seeds) * n_splits
    pbar = tqdm(total=total_folds, desc="Total Folds Progress")
    
    tasks = []
    for dataset in args.datasets:
        for seed in args.seeds:
            if IS_HPC:
                # HPC: fold-level parallelism to saturate many cores
                for fold in range(1, n_splits + 1):
                    tasks.append({'dataset': dataset, 'seed': seed, 'pbar': pbar, 'n_splits': n_splits, 'fold': fold})
            else:
                # Local: seed-level parallelism (Original behavior)
                tasks.append({'dataset': dataset, 'seed': seed, 'pbar': pbar, 'n_splits': n_splits, 'fold': -1})
            
    print("="*60)
    print("🚀 STARTING EXPERIMENT ORCHESTRATION")
    print("="*60)
    print(f"📌 Datasets: {args.datasets}")
    print(f"📌 Seeds: {args.seeds}")
    print(f"⚡ Max Workers: {args.max_workers} (Auto-detected HPC: {'Yes' if IS_HPC else 'No, running Local Mode'})")
    print(f"📋 Total sequence tasks: {len(tasks)}")
    print(f"📋 Total individual folds to run: {total_folds}")
    print(" Output will be hidden and written to logs/")
    print("="*60 + "\n")
    
    results = []
    # Using ThreadPoolExecutor for lightweight subprocess management
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                print(res)
                results.append(res)
            except Exception as e:
                print(f"[!!!] Process exception: {e}")
                
    pbar.close()
    print("\n" + "="*60)
    print("✅ ALL TASKS COMPLETED")
    print("="*60)
