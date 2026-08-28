import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
RESULTS_DIR = os.environ.get('RESULTS_DIR', os.path.join(BASE_DIR, "Results"))

os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = {
    "Plant_oil": "combined_oil_dataset.csv",
    "Brewed_vinegar": "combined_vinegar_dataset.csv",
    "Wine_spoilage": "combined_wine_dataset.csv",
    "Chinese_wine": "combined_chinese_wine_dataset.csv",
    "Coffee": "combined_coffee_dataset.csv"
}

N_SPLITS = 10  # 10-fold cross-validation (10% test per fold)
N_SEED_SETS = 20  # Choose 1->20 measurement set per label as generation seed set

N_ITER = 20  # 20-combination (Test 20 random hyperparameter combinations)

# --- WORKSTATION SETTINGS ---
# When running on personal i7, keep N_JOBS = 3 (default) to prevent freezing
# When running on Workstation (64-128 cores), set N_JOBS = -1 to use all cores
N_JOBS = int(os.environ.get('N_JOBS', 3))

# Disable GPU by default to prevent Cuda/MPS errors on personal Windows machine
# Set to True when deploying to Workstation with powerful NVIDIA GPU
USE_GPU = False

RANDOM_STATE = 42
