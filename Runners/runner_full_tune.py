import os
import subprocess
import concurrent.futures
import sys
import argparse
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import N_SPLITS

EXPERIMENT_SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'experiment_full_tune.py')
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs', 'full_tune')

IS_HPC = "SLURM_JOB_ID" in os.environ

if IS_HPC:
    cpus = os.environ.get("SLURM_CPUS_PER_TASK", "1")
    os.environ["OMP_NUM_THREADS"] = cpus
    os.environ["OPENBLAS_NUM_THREADS"] = cpus
    os.environ["MKL_NUM_THREADS"] = cpus
    os.environ["VECLIB_MAXIMUM_THREADS"] = cpus
    os.environ["NUMEXPR_NUM_THREADS"] = cpus
    os.environ["N_JOBS"] = cpus 
else:
    os.environ["N_JOBS"] = "-1"

def build_tasks():
    datasets = {
        "Plant_oil": [1, 2, 5, 10, 20],
        "Brewed_vinegar": [1, 2, 5, 10, 20],
        "Wine_spoilage": [1, 2, 5, 10, 20],
        "Chinese_wine": [1, 2, 5, 10, 20],
        "Coffee": [1, 2, 5, 10, 20]
    }
    
    generators = ['Jittering', 'Scaling', 'Magnitude_Warping', 'Time_Warping', 
                  'SMOTE', 'ADASYN', 'Mixup', 'GMM', 'HMM_GMM', 'TimeVAE']
    
    scale_factors = [2, 3, 5, 10]
    
    tasks = []
    
    for dataset, seeds in datasets.items():
        for seed in seeds:
            for generator in generators:
                # Apply rules from experiment.py
                if generator in ['SMOTE', 'Mixup', 'GMM', 'HMM_GMM'] and seed < 3:
                    continue
                if generator == 'ADASYN' and seed < 10:
                    continue
                if generator in ['TimeVAE'] and seed <= 10:
                    continue
                
                tasks.append({
                    'dataset': dataset,
                    'seed': seed,
                    'generator': generator,
                    'n_splits': N_SPLITS
                })
    
    # SORTING: Reverse the tasks so that the heaviest jobs (Seed 20, TimeVAE, etc.) 
    # are dispatched first. This prevents SLURM from getting stuck with long jobs at the end.
    tasks.reverse()
    return tasks

def run_task(task_kwargs, pbar=None):
    dataset = task_kwargs['dataset']
    seed = task_kwargs['seed']
    generator = task_kwargs['generator']
    n_splits = task_kwargs.get('n_splits', 10)
    
    log_file_name = f"{dataset}_S{seed}_{generator}_AllScales.log"
    log_file = os.path.join(LOG_DIR, log_file_name)
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Starting Full Tune: {dataset} - Seed {seed} - Gen {generator} - All Scales\n\n")
        f.flush()
        
        # Sequentially run all folds for this config
        for fold in range(1, n_splits + 1):
            command = [
                sys.executable, EXPERIMENT_SCRIPT, 
                "--dataset", dataset, 
                "--seed", str(seed), 
                "--fold", str(fold),
                "--generator", generator,
                "--scale_factor", "-1"
            ]
            process = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, text=True)
            process.wait()
            
            if process.returncode != 0:
                return f"[!] ERROR in {dataset} (Seed {seed}, Gen {generator}, Fold {fold}) - Log: {log_file}"
                
        if pbar:
            pbar.update(1)
                
    return f"[+] COMPLETED Full Tune: {dataset} (Seed {seed}, Gen {generator})"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--get_task_count", action="store_true", help="Print the total number of valid tasks and exit")
    args = parser.parse_args()
    
    tasks = build_tasks()
    
    if args.get_task_count:
        print(len(tasks))
        sys.exit(0)
        
    os.makedirs(LOG_DIR, exist_ok=True)
    
    if IS_HPC:
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task_id is not None:
            task_id = int(task_id)
            if task_id < len(tasks):
                print(f"Running task {task_id} / {len(tasks)}")
                res = run_task(tasks[task_id])
                print(res)
            else:
                print(f"Task ID {task_id} out of range (max {len(tasks)-1})")
        else:
            print("SLURM_ARRAY_TASK_ID not found but IS_HPC is true.")
    else:
        print(f"Running {len(tasks)} tasks locally...")
        pbar = tqdm(total=len(tasks), desc="Total Configs Progress")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for t in tasks:
                futures.append(executor.submit(run_task, t, pbar))
                
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    # print(res)
                except Exception as e:
                    print(f"[!!!] Process exception: {e}")
        pbar.close()
