import os
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from config import DATASET_DIR, DATASETS, N_SPLITS, RANDOM_STATE

def load_dataset(dataset_name):
    file_path = os.path.join(DATASET_DIR, DATASETS[dataset_name])
    df = pd.read_csv(file_path)
    
    # Normalize Time column name
    for col in df.columns:
        if 'Time' in col and col != 'Time':
            df.rename(columns={col: 'Time'}, inplace=True)
            break
            
    return df

def get_feature_columns(df):
    exclude_cols = ['Measurement_Number', 'Time', 'Label']
    return [col for col in df.columns if col not in exclude_cols]

def get_train_test_splits(df):
    """
    Generator yielding (train_idx, test_idx) for N-fold cross-validation.
    Ensures that data with the same measurement number do not leak outside the test set,
    and each fold gets a stratified proportion of measurements per label.
   
    """
    groups = df['Measurement_Number'].values
    y = df['Label'].values
    
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    for train_idx, test_idx in sgkf.split(df, y, groups=groups):
        yield train_idx, test_idx

def get_generation_seed_sets(df, train_idx, num_sets=20, random_seed_offset=0):
    """
    From the 90% training data, choose `num_sets` measurement sets per label as the generation seed set.
    Returns indices of the seed set.
   
    """
    train_df = df.iloc[train_idx]
    
    seed_indices = []
    labels = sorted(train_df['Label'].unique())
    
    for label in labels:
        # Reset seed per label to ensure perfectly nested subsets
        # (e.g. elements of seed_size=2 are always included in seed_size=5)
        #
        try:
            lbl_seed = int(label)
        except ValueError:
            lbl_seed = sum(ord(c) for c in str(label))
            
        np.random.seed(RANDOM_STATE + lbl_seed + random_seed_offset)
        
        label_df = train_df[train_df['Label'] == label]
        unique_measurements = sorted(label_df['Measurement_Number'].unique())
        
        # If there are fewer measurements than requested, take all available
        #
        n_samples = min(num_sets, len(unique_measurements))
        
        if n_samples > 0:
            chosen_measurements = np.random.choice(unique_measurements, n_samples, replace=False)
            
            idx = label_df[label_df['Measurement_Number'].isin(chosen_measurements)].index.tolist()
            seed_indices.extend(idx)
            
    return seed_indices
