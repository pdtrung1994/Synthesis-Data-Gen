import os
import subprocess
import concurrent.futures
import sys
import argparse
from tqdm import tqdm

# Configure paths
EXPERIMENT_SCRIPT = os.path.join('..', 'experiment_gpu_short_fulltune.py')
LOG_DIR = 'logs'

def run_task(task_kwargs):
    dataset = task_kwargs['dataset']
    seed = task_kwargs['seed']
    pbar = task_kwargs['pbar']
    n_splits = task_kwargs.get('n_splits', 10)
    log_file = os.path.join(LOG_DIR, f"{dataset}_S{seed}.log")
    
    # Open log file to write output directly
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Starting {dataset} - Seed {seed}\n\n")
        f.flush()
        
        for fold in range(1, n_splits + 1):
            command = [sys.executable, EXPERIMENT_SCRIPT, "--dataset", dataset, "--seed", str(seed), "--fold", str(fold)]
            process = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, text=True)
            process.wait()
            
            pbar.update(1)
            
            if process.returncode != 0:
                return f"[!] ERROR in {dataset} (Seed {seed}, Fold {fold}) - See details at {log_file}"
                
    return f"[+] COMPLETED {dataset} (Seed {seed})"

if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    
    DEFAULT_DATASETS = ["Plant_oil", "Brewed_vinegar", "Wine_spoilage", "Chinese_wine", "Coffee"]
    DEFAULT_SEEDS = [1, 2, 5, 10, 20]
    
    parser = argparse.ArgumentParser(description="Multi-process Orchestrator for Experiments")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, 
                        help="List of datasets to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="List of seeds to run")
    parser.add_argument("--max_workers", type=int, default=6,
                        help="Maximum number of parallel workers (default: 6)")
                        
    args = parser.parse_args()
    
    n_splits = 10
    total_folds = len(args.datasets) * len(args.seeds) * n_splits
    pbar = tqdm(total=total_folds, desc="Total Folds Progress")
    
    tasks = []
    for dataset in args.datasets:
        for seed in args.seeds:
            tasks.append({'dataset': dataset, 'seed': seed, 'pbar': pbar, 'n_splits': n_splits})
            
    print("="*50)
    print("🚀 STARTING EXPERIMENT ORCHESTRATION")
    print("="*50)
    print(f"📌 Datasets: {args.datasets}")
    print(f"📌 Seeds: {args.seeds}")
    print(f"⚡ Max Workers: {args.max_workers}")
    print(f"📋 Total sequence tasks: {len(tasks)}")
    print(f"📋 Total individual folds to run: {total_folds}")
    print(" Output will be hidden and written to Runners/logs/")
    print("="*50 + "\n")
    
    results = []
    # Use ThreadPoolExecutor instead of ProcessPool for absolute Windows compatibility
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
    print("\n" + "="*50)
    print("✅ ALL TASKS COMPLETED")
    print("="*50)
