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
from models import get_models_and_params, tune_hyperparameters
import augmentations
from utils import plot_performance_comparison, save_results_table, append_result_to_csv

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

def run_experiment(dataset_name, target_fold=-1, target_seed=-1, target_selections=10):
    print(f"\n{'='*50}\nRunning experiment for {dataset_name}\n{'='*50}")
    
    run_id = dataset_name
    if target_fold != -1: run_id += f"_F{target_fold}"
    if target_seed != -1: run_id += f"_S{target_seed}"
    
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
                            'Model': model_name, 'Method': 'Baseline', 'Accuracy': acc_base,
                            'F1_Score': f1_base, 'Hyperparameters': params_str, 'ExecutionTime_Sec': exec_time
                        }
                        all_results.append(res_dict)
                        append_result_to_csv(res_dict, run_id)
                        print(f"      -> [Baseline] Acc: {acc_base:.4f}, F1: {f1_base:.4f}, Time: {exec_time:.2f}s")
                    except Exception as e:
                        print(f"      [!] Baseline failed for {model_name}: {e}")
                
                    # --- 2. HP Tuning ---
                    if not is_dl_model:
                        print(f"      HP Tuning {model_name}...")
                        try:
                            cv_folds = min(5, n_seeds) if n_seeds >= 2 else 2
                            if n_seeds < 2:
                                # Fallback to standard CV (may leak) when there's not enough measurements for grouped CV
                                best_model, best_params = tune_hyperparameters(
                                    base_model, params, X_seed_model, y_seed_model, 
                                    groups=None, n_iter=N_ITER, cv=2
                                )
                            else:
                                best_model, best_params = tune_hyperparameters(
                                    base_model, params, X_seed_model, y_seed_model, 
                                    groups=groups_seed_model, n_iter=N_ITER, cv=cv_folds
                                )
                        except Exception as e:
                            print(f"      [!] HP Tuning failed, using base model: {e}")
                            best_model = base_model
                    else:
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
                            'Model': model_name, 'Method': 'Tuned_Baseline', 'Accuracy': acc_tuned,
                            'F1_Score': f1_tuned, 'Hyperparameters': params_str, 'ExecutionTime_Sec': exec_time
                        }
                        all_results.append(res_dict)
                        append_result_to_csv(res_dict, run_id)
                        print(f"      -> [Tuned_Baseline] Acc: {acc_tuned:.4f}, F1: {f1_tuned:.4f}, Time: {exec_time:.2f}s")
                    except Exception as e:
                        print(f"      [!] Tuned Baseline failed for {model_name}: {e}")
                
                # --- 3. Generation Methods ---
                generation_methods = {
                    'Jittering': lambda X, y: (augmentations.jitter(X), y),
                    'Scaling': lambda X, y: (augmentations.scaling(X), y),
                    'Magnitude_Warping': lambda X, y: (augmentations.magnitude_warping(X), y),
                    'Time_Warping': lambda X, y: (augmentations.time_warping(X), y),
                    'SMOTE': augmentations.apply_smote,
                    'ADASYN': augmentations.apply_adasyn,
                    'Mixup': augmentations.mixup,
                    'GMM': augmentations.apply_gmm,
                    'HMM_GMM': augmentations.apply_hmm_gmm,
                    'TimeVAE': augmentations.apply_timevae
                }
            
                for gen_name, gen_func in generation_methods.items():
                    # Condition 1: Advanced methods require at least 3 seeds to work properly
                    if gen_name in ['SMOTE', 'Mixup', 'GMM', 'HMM_GMM'] and n_seeds < 3:
                        continue
                    # Condition 2: ADASYN requires at least 10 seeds to avoid mathematical errors
                    if gen_name == 'ADASYN' and n_seeds < 10:
                        continue
                    # Condition 3: TimeVAE only when seed size > 10
                    if gen_name in ['TimeVAE'] and n_seeds <= 10:
                        continue
                
                    for scale_factor in [2, 3, 5, 10]:
                        print(f"    [=== Generating {scale_factor}x synthetic data with {gen_name} ===]")
                        try:
                            # Methods that handle scaling internally
                            if gen_name in ['SMOTE', 'ADASYN', 'GMM', 'HMM_GMM', 'TimeVAE']:
                                X_syn, y_syn = gen_func(X_seed, y_seed, scale_factor=scale_factor)  # type: ignore
                            else:
                                # Methods that return ONLY synthetic data of size 1x
                                X_syn_list = [X_seed]
                                y_syn_list = [y_seed]
                                for _ in range(scale_factor - 1):
                                    X_aug, y_aug = gen_func(X_seed, y_seed)
                                    X_syn_list.append(X_aug)
                                    y_syn_list.append(y_aug)
                                X_syn = np.vstack(X_syn_list)
                                y_syn = np.hstack(y_syn_list)
                        except Exception as e:
                            print(f"      [!] Failed generation with {gen_name} at {scale_factor}x: {e}")
                            continue
                            
                        # --- 4. Evaluate Synthetic dataset ---
                        for model_name, best_model in tuned_models.items():
                            is_dl_model = is_dl_models[model_name]
                            
                            if not is_dl_model:
                                X_syn_model = X_syn.reshape(-1, X_syn.shape[2])
                                y_syn_model = np.repeat(y_syn, X_syn.shape[1])
                                X_test_model = X_test.reshape(-1, X_test.shape[2])
                                y_test_model = np.repeat(y_test, X_test.shape[1])
                            else:
                                X_syn_model = X_syn
                                y_syn_model = y_syn
                                X_test_model = X_test
                                y_test_model = y_test
                            
                            # Limit training rows to 100,000 for speed
                            MAX_ROWS = 100000
                            if X_syn_model.shape[0] > MAX_ROWS:
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
                                    'Model': model_name, 'Method': gen_name, 'Accuracy': acc_syn,
                                    'F1_Score': f1_syn, 'Hyperparameters': params_str, 'ExecutionTime_Sec': exec_time
                                }
                                all_results.append(res_dict)
                                append_result_to_csv(res_dict, run_id)
                                print(f"      -> [{model_name} + {gen_name} @ {scale_factor}x] Acc: {acc_syn:.4f}, F1: {f1_syn:.4f}, Time: {exec_time:.2f}s")
                            except Exception as e:
                                print(f"      [!] Failed evaluation with {model_name} on {gen_name} at {scale_factor}x: {e}")
                        
        fold_idx += 1
        
    results_df = pd.DataFrame(all_results)
    # save_results_table() removed here because append_result_to_csv safely saves row by row
    
    if not results_df.empty:
        plot_df = results_df[(results_df['SeedsPerLabel'] == 20) & (results_df['ScaleFactor'] == 10)].groupby(['Method', 'Model'])['Accuracy'].mean().reset_index()
        plot_performance_comparison(plot_df, run_id)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Synthetic Data Generation Experiments")
    parser.add_argument("--dataset", type=str, default="All", help="Dataset name to run, or 'All'")
    parser.add_argument("--fold", type=int, default=-1, help="Target fold to run (e.g. 1). Default is -1 (all folds)")
    parser.add_argument("--seed", type=int, default=-1, help="Target seed size to run (e.g. 5). Default is -1 (all seed sizes)")
    parser.add_argument("--n_selections", type=int, default=10, help="Number of seed fold selections to run. Default is 10.")
    args = parser.parse_args()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    datasets_to_run = DATASETS.keys() if args.dataset == "All" else [args.dataset]
    
    for d_name in datasets_to_run:
        if d_name not in DATASETS:
            print(f"Warning: Dataset '{d_name}' not found in config. Skipping.")
            continue
            
        run_experiment(d_name, target_fold=args.fold, target_seed=args.seed, target_selections=args.n_selections)
