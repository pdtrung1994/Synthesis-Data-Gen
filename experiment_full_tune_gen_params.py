import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import traceback
import warnings
import time
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from config import DATASETS, N_SEED_SETS, N_ITER, RESULTS_DIR, N_SPLITS, RANDOM_STATE
from dataset import load_dataset, get_feature_columns, get_train_test_splits, get_generation_seed_sets
from models import get_models_and_params, tune_hyperparameters, tune_hyperparameters_custom_val
import augmentations
from utils import plot_performance_comparison, save_results_table, append_result_to_csv

# Ensure reproducibility for random search combinations
np.random.seed(42)

# --- Define Generation Parameter Grid ---
GEN_PARAMS_GRID = {
    'Jittering': [{'sigma': s} for s in [0.01, 0.03, 0.05, 0.1, 0.15]],
    'Scaling': [{'sigma': s} for s in [0.05, 0.1, 0.2, 0.3, 0.5]],
    'Magnitude_Warping': [{'sigma': float(s), 'knot': int(k)} for s, k in zip(
        np.random.choice([0.05, 0.1, 0.2, 0.3, 0.5], 10),
        np.random.choice([3, 4, 5, 6, 8], 10)
    )],
    'Time_Warping': [{'sigma': float(s), 'knot': int(k)} for s, k in zip(
        np.random.choice([0.05, 0.1, 0.2, 0.3, 0.5], 10),
        np.random.choice([3, 4, 5, 6, 8], 10)
    )],
    'Mixup': [{'alpha': a} for a in [0.1, 0.2, 0.4, 0.6, 0.8]],
    'SMOTE': [{'k_neighbors': k} for k in [1, 2, 3, 4, 5]],
    'ADASYN': [{'n_neighbors': k} for k in [1, 2, 3, 4, 5]],
    'GMM': [{'n_components': int(n), 'covariance_type': str(c)} for n, c in zip(
        np.random.choice([1, 2, 3, 4, 5], 10),
        np.random.choice(['full', 'tied', 'diag', 'spherical'], 10)
    )],
    'HMM_GMM': [{'n_components': int(n), 'n_mix': int(m), 'covariance_type': str(c)} for n, m, c in zip(
        np.random.choice([1, 2, 3, 4, 5], 10),
        np.random.choice([1, 2, 3, 4, 5], 10),
        np.random.choice(['full', 'tied', 'diag', 'spherical'], 10)
    )],
    'TimeVAE': [{'latent_dim': int(l), 'epochs': int(e), 'batch_size': int(b)} for l, e, b in zip(
        np.random.choice([8, 16, 24, 32, 64], 10),
        np.random.choice([20, 30, 40, 50, 60], 10),
        np.random.choice([16, 32, 64, 128], 10)
    )]
}

# Reset seed
np.random.seed(RANDOM_STATE)

def extract_timeseries_features(df, feature_cols, target_timesteps=100):
    """
    Helper function to reshape dataframe into (samples, timesteps, features) and optionally resample.
    """
    grouped = df.groupby('Measurement_Number')
    
    X = []
    y = []
    for name, group in grouped:
        group = group.sort_values('Time').copy()
        
        # Robustly impute missing values (NaN): forward fill -> backward fill -> fallback to 0
        group[feature_cols] = group[feature_cols].ffill().bfill().fillna(0)
        
        features = group[feature_cols].values
        
        if target_timesteps is not None:
            orig_len = features.shape[0]
            if orig_len != target_timesteps:
                features_new = np.zeros((target_timesteps, features.shape[1]))
                for j in range(features.shape[1]):
                    features_new[:, j] = np.interp(
                        np.linspace(0, 1, target_timesteps),
                        np.linspace(0, 1, orig_len),
                        features[:, j]
                    )
                features = features_new
                
        label = group['Label'].iloc[0]
        
        X.append(features)
        y.append(label)
        
    return np.array(X), np.array(y)

def run_experiment(dataset_name, target_fold=-1, target_seed=-1, target_selections=10, target_generator="All", target_scale=-1):
    print(f"\n{'='*50}\nRunning gen-params experiment for {dataset_name}\n{'='*50}")
    
    run_id = dataset_name + "_GenParams"
    if target_fold != -1: run_id += f"_F{target_fold}"
    if target_seed != -1: run_id += f"_S{target_seed}"
    if target_generator != "All": run_id += f"_{target_generator}"
    if target_scale != -1: run_id += f"_Scale{target_scale}"
    
    # Remove old result file to prevent duplicate appending if rerun
    csv_path = os.path.join(RESULTS_DIR, f"{run_id}_results.csv")
    if os.path.exists(csv_path):
        try:
            os.remove(csv_path)
            print(f"[!] Deleted old result file: {csv_path}")
        except Exception as e:
            print(f"[!] Could not delete old result file {csv_path}: {e}")
            
    try:
        df = load_dataset(dataset_name)
    except Exception as e:
        print(f"Could not load {dataset_name}: {e}")
        return
        
    feature_cols = get_feature_columns(df)
    all_results = []
    
    fold_idx = 1
    for full_train_idx, test_idx in get_train_test_splits(df):
        if target_fold == -1 and fold_idx > 10:
            break
            
        if target_fold != -1 and fold_idx != target_fold:
            fold_idx += 1
            continue
            
        print(f"\n--- Fold {fold_idx}/{N_SPLITS} ---")
        
        test_df = df.iloc[test_idx]
        X_test_raw, y_test = extract_timeseries_features(test_df, feature_cols)
        
        for n_seeds in [1, 2, 5, 10, 20]:
            if target_seed != -1 and n_seeds != target_seed:
                continue
                
            print(f"\n  >> Seed sets per label: {n_seeds}")
            for seed_selection_idx in range(1, target_selections + 1):
                print(f"\n    >> Seed Fold Selection: {seed_selection_idx}/10")
                seed_indices = get_generation_seed_sets(df, full_train_idx, num_sets=n_seeds, random_seed_offset=seed_selection_idx)
                seed_df = df.iloc[seed_indices]
            
                X_seed_raw, y_seed = extract_timeseries_features(seed_df, feature_cols)
            
                # Standardize features based on the current training seed set to prevent data leakage
                n_timesteps = X_seed_raw.shape[1]
                n_features = X_seed_raw.shape[2]
            
                scaler = StandardScaler()
            
                X_seed_2d = X_seed_raw.reshape(-1, n_features)
                X_seed_scaled = scaler.fit_transform(X_seed_2d)
                X_seed = X_seed_scaled.reshape(-1, n_timesteps, n_features)
            
                X_test_2d = X_test_raw.reshape(-1, n_features)
                X_test_scaled = scaler.transform(X_test_2d)
                X_test = X_test_scaled.reshape(-1, n_timesteps, n_features)
            
                models_dict = get_models_and_params()
                tuned_models = {}
                is_dl_models = {}
            
                for model_name, model_info in models_dict.items():
                    print(f"    Evaluating Model: {model_name}")
                    base_model = model_info['model']
                    params = model_info['params']
                
                    is_dl_model = model_name in ['GRU', 'RNN']
                    is_dl_models[model_name] = is_dl_model
                
                    # Prepare point-level data for Scikit-Learn models vs sequence-level for DL
                    if not is_dl_model:
                        X_seed_model = X_seed.reshape(-1, X_seed.shape[2])
                        y_seed_model = np.repeat(y_seed, X_seed.shape[1])
                        groups_seed_model = np.repeat(np.arange(X_seed.shape[0]), X_seed.shape[1])
                        X_test_model = X_test.reshape(-1, X_test.shape[2])
                        y_test_model = np.repeat(y_test, X_test.shape[1])
                    else:
                        X_seed_model = X_seed
                        y_seed_model = y_seed
                        groups_seed_model = None
                        X_test_model = X_test
                        y_test_model = y_test
                
                    # --- 1. Baseline ---
                    try:
                        start_time = time.time()
                        base_model.fit(X_seed_model, y_seed_model)
                        y_pred_base = base_model.predict(X_test_model)
                        exec_time = time.time() - start_time
                    
                        acc_base = accuracy_score(y_test_model, y_pred_base)
                        f1_base = f1_score(y_test_model, y_pred_base, average='weighted')
                        params_str = str(base_model.get_params()) if hasattr(base_model, 'get_params') else "N/A"
                        res_dict = {
                            'Dataset': dataset_name, 'Fold': fold_idx, 'SeedsPerLabel': n_seeds, 'SeedFoldSelection': seed_selection_idx,
                            'ScaleFactor': 1,
                            'Model': model_name, 'Method': 'Baseline', 'GenParamSetIndex': 'N/A', 'GenParams': 'N/A',
                            'Accuracy': acc_base,
                            'F1_Score': f1_base, 'Hyperparameters': params_str, 'ExecutionTime_Sec': exec_time
                        }
                        all_results.append(res_dict)
                        append_result_to_csv(res_dict, run_id)
                        print(f"      -> [Baseline] Acc: {acc_base:.4f}, F1: {f1_base:.4f}, Time: {exec_time:.2f}s")
                    except Exception as e:
                        print(f"      [!] Baseline failed for {model_name}: {e}")
                
                    # --- 2. HP Tuning ---
                    print(f"      HP Tuning {model_name}...")
                    try:
                        cv_folds = min(5, n_seeds) if n_seeds >= 2 else 2
                        groups_to_use = groups_seed_model if (n_seeds >= 2 and not is_dl_model) else None
                        
                        best_model, best_params = tune_hyperparameters(
                            base_model, params, X_seed_model, y_seed_model, 
                            groups=groups_to_use, n_iter=N_ITER, cv=cv_folds if n_seeds >= 2 else 2
                        )
                    except Exception as e:
                        print(f"      [!] HP Tuning failed, using base model: {e}")
                        best_model = base_model
                    
                    tuned_models[model_name] = best_model
                    
                    # --- 2.5 Tuned Baseline ---
                    try:
                        start_time = time.time()
                        y_pred_tuned = best_model.predict(X_test_model)
                        exec_time = time.time() - start_time
                    
                        acc_tuned = accuracy_score(y_test_model, y_pred_tuned)
                        f1_tuned = f1_score(y_test_model, y_pred_tuned, average='weighted')
                        params_str = str(best_model.get_params()) if hasattr(best_model, 'get_params') else "N/A"
                        res_dict = {
                            'Dataset': dataset_name, 'Fold': fold_idx, 'SeedsPerLabel': n_seeds, 'SeedFoldSelection': seed_selection_idx,
                            'ScaleFactor': 1,
                            'Model': model_name, 'Method': 'Tuned_Baseline', 'GenParamSetIndex': 'N/A', 'GenParams': 'N/A', 
                            'Accuracy': acc_tuned,
                            'F1_Score': f1_tuned, 'Hyperparameters': params_str, 'ExecutionTime_Sec': exec_time
                        }
                        all_results.append(res_dict)
                        append_result_to_csv(res_dict, run_id)
                        print(f"      -> [Tuned_Baseline] Acc: {acc_tuned:.4f}, F1: {f1_tuned:.4f}, Time: {exec_time:.2f}s")
                    except Exception as e:
                        print(f"      [!] Tuned Baseline failed for {model_name}: {e}")
                
                # --- 3. Generation Methods ---
                generation_methods = {
                    'Jittering': lambda X, y, **kwargs: (augmentations.jitter(X, **kwargs), y),
                    'Scaling': lambda X, y, **kwargs: (augmentations.scaling(X, **kwargs), y),
                    'Magnitude_Warping': lambda X, y, **kwargs: (augmentations.magnitude_warping(X, **kwargs), y),
                    'Time_Warping': lambda X, y, **kwargs: (augmentations.time_warping(X, **kwargs), y),
                    'SMOTE': augmentations.apply_smote,
                    'ADASYN': augmentations.apply_adasyn,
                    'Mixup': augmentations.mixup,
                    'GMM': augmentations.apply_gmm,
                    'HMM_GMM': augmentations.apply_hmm_gmm,
                    'TimeVAE': augmentations.apply_timevae
                }
            
                for gen_name, gen_func in generation_methods.items():
                    if target_generator != "All" and gen_name != target_generator:
                        continue
                    # Condition 1: Advanced methods require at least 3 seeds to work properly
                    if gen_name in ['SMOTE', 'Mixup', 'GMM', 'HMM_GMM'] and n_seeds < 3:
                        continue
                    # Condition 2: ADASYN requires at least 10 seeds to avoid mathematical errors
                    if gen_name == 'ADASYN' and n_seeds < 10:
                        continue
                    # Condition 3: TimeVAE only when seed size > 10
                    if gen_name in ['TimeVAE'] and n_seeds <= 10:
                        continue
                
                    scale_factors_to_run = [2, 3, 5, 10] if target_scale == -1 else [target_scale]
                    for scale_factor in scale_factors_to_run:
                        # NEW: Iterate over parameter sets
                        param_sets = GEN_PARAMS_GRID.get(gen_name, [{}])
                        for param_idx, kwargs in enumerate(param_sets):
                            print(f"    [=== Generating {scale_factor}x synthetic data with {gen_name} (Set {param_idx + 1}, Params: {kwargs}) ===]")
                            try:
                                # Methods that handle scaling internally
                                if gen_name in ['SMOTE', 'ADASYN', 'GMM', 'HMM_GMM', 'TimeVAE']:
                                    X_syn, y_syn = gen_func(X_seed, y_seed, scale_factor=scale_factor, **kwargs)  # type: ignore
                                else:
                                    # Methods that return ONLY synthetic data of size 1x
                                    X_syn_list = [X_seed]
                                    y_syn_list = [y_seed]
                                    for _ in range(scale_factor - 1):
                                        X_aug, y_aug = gen_func(X_seed, y_seed, **kwargs)
                                        X_syn_list.append(X_aug)
                                        y_syn_list.append(y_aug)
                                    X_syn = np.vstack(X_syn_list)
                                    y_syn = np.hstack(y_syn_list)
                            except Exception as e:
                                print(f"      [!] Failed generation with {gen_name} at {scale_factor}x: {e}")
                                continue
                                
                            # --- 4. Evaluate Synthetic dataset ---
                            for model_name, model_info in models_dict.items():
                                base_model = model_info['model']
                                params = model_info['params']
                                is_dl_model = is_dl_models[model_name]
                                MAX_ROWS = 100000
                                
                                if not is_dl_model:
                                    X_syn_model = X_syn.reshape(-1, X_syn.shape[2])
                                    y_syn_model = np.repeat(y_syn, X_syn.shape[1])
                                    X_test_model = X_test.reshape(-1, X_test.shape[2])
                                    y_test_model = np.repeat(y_test, X_test.shape[1])
                                    
                                    if X_syn_model.shape[0] > MAX_ROWS:
                                        np.random.seed(RANDOM_STATE)
                                        indices = np.random.choice(X_syn_model.shape[0], MAX_ROWS, replace=False)
                                        X_syn_model = X_syn_model[indices]
                                        y_syn_model = y_syn_model[indices]
                                    
                                    X_seed_model_val = X_seed.reshape(-1, X_seed.shape[2])
                                    y_seed_model_val = np.repeat(y_seed, X_seed.shape[1])
                                    
                                    try:
                                        best_model, best_params = tune_hyperparameters_custom_val(
                                            base_model, params, X_syn_model, y_syn_model, X_seed_model_val, y_seed_model_val, n_iter=N_ITER
                                        )
                                    except Exception as e:
                                        best_model = base_model
                                else:
                                    X_syn_model = X_syn
                                    y_syn_model = y_syn
                                    X_test_model = X_test
                                    y_test_model = y_test
                                    
                                    import sklearn
                                    # Clone the baseline-tuned model to reset its weights
                                    best_model = sklearn.base.clone(tuned_models[model_name])
                                
                                # MAX_ROWS optimization already applied before tuning for scikit-learn models
                                if is_dl_model and X_syn_model.shape[0] > MAX_ROWS:
                                    np.random.seed(RANDOM_STATE)
                                    indices = np.random.choice(X_syn_model.shape[0], MAX_ROWS, replace=False)
                                    X_syn_model = X_syn_model[indices]
                                    y_syn_model = y_syn_model[indices]
                                    
                                try:
                                    start_time = time.time()
                                    best_model.fit(X_syn_model, y_syn_model)
                                    y_pred_syn = best_model.predict(X_test_model)
                                    exec_time = time.time() - start_time
                                
                                    acc_syn = accuracy_score(y_test_model, y_pred_syn)
                                    f1_syn = f1_score(y_test_model, y_pred_syn, average='weighted')
                                    params_str = str(best_model.get_params()) if hasattr(best_model, 'get_params') else "N/A"
                                
                                    res_dict = {
                                        'Dataset': dataset_name, 'Fold': fold_idx, 'SeedsPerLabel': n_seeds, 'SeedFoldSelection': seed_selection_idx,
                                        'ScaleFactor': scale_factor,
                                        'Model': model_name, 'Method': gen_name, 
                                        'GenParamSetIndex': param_idx + 1, 'GenParams': str(kwargs),
                                        'Accuracy': acc_syn,
                                        'F1_Score': f1_syn, 'Hyperparameters': params_str, 'ExecutionTime_Sec': exec_time
                                    }
                                    all_results.append(res_dict)
                                    append_result_to_csv(res_dict, run_id)
                                    print(f"      -> [{model_name} + {gen_name} Set {param_idx + 1}] Acc: {acc_syn:.4f}, F1: {f1_syn:.4f}, Time: {exec_time:.2f}s")
                                except Exception as e:
                                    print(f"      [!] Failed evaluation with {model_name} on {gen_name} Set {param_idx + 1}: {e}")
                        
        fold_idx += 1
        
    results_df = pd.DataFrame(all_results)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Synthetic Data Generation Parameter Experiments")
    parser.add_argument("--dataset", type=str, default="All", help="Dataset name to run, or 'All'")
    parser.add_argument("--fold", type=int, default=-1, help="Target fold to run (e.g. 1). Default is -1 (all folds)")
    parser.add_argument("--seed", type=int, default=-1, help="Target seed size to run (e.g. 5). Default is -1 (all seed sizes)")
    parser.add_argument("--n_selections", type=int, default=10, help="Number of seed fold selections to run. Default is 10.")
    parser.add_argument("--generator", type=str, default="All", help="Target generator method. Default is All.")
    parser.add_argument("--scale_factor", type=int, default=-1, help="Target scale factor. Default is -1 (all scale factors).")
    args = parser.parse_args()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    datasets_to_run = DATASETS.keys() if args.dataset == "All" else [args.dataset]
    
    for d_name in datasets_to_run:
        if d_name not in DATASETS:
            print(f"Warning: Dataset '{d_name}' not found in config. Skipping.")
            continue
            
        run_experiment(d_name, target_fold=args.fold, target_seed=args.seed, target_selections=args.n_selections, target_generator=args.generator, target_scale=args.scale_factor)
