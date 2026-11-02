import os
# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (
    HalvingRandomSearchCV,
    RandomizedSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    PredefinedSplit,
)
from config import RANDOM_STATE, USE_GPU, N_JOBS

try:
    # pyrefly: ignore [missing-import]
    import torch
    # pyrefly: ignore [missing-import]
    import torch.nn as nn
    # pyrefly: ignore [missing-import]
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

class PyTorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(PyTorchRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

class DeepLearningBaseClassifier(ClassifierMixin, BaseEstimator):
    """
    PyTorch RNN Classifier wrapper acting like a scikit-learn estimator
   
    """
    def __init__(self, hidden_size=64, epochs=10, batch_size=32, lr=0.001):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model = None
        if not PYTORCH_AVAILABLE:
            print("PyTorch not installed. Skipping training.")
            return self
            
        self.class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([self.class_to_idx[c] for c in y])
        
        if USE_GPU:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
            torch.set_flush_denormal(True)
            torch.set_num_threads(1)

        torch.manual_seed(RANDOM_STATE)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RANDOM_STATE)
            
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y_idx)
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Handle small batch sizes
        current_batch = min(self.batch_size, len(X))
        if current_batch < 1: current_batch = 1
        dataloader = DataLoader(dataset, batch_size=current_batch, shuffle=True)
        
        input_size = X.shape[2]
        num_classes = len(self.classes_)
        
        self.model = PyTorchRNN(input_size, self.hidden_size, num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        return self
        
    def predict(self, X):
        classes_ = getattr(self, "classes_", None)
        model = getattr(self, "model", None)
        if not PYTORCH_AVAILABLE or model is None or classes_ is None or len(classes_) == 0:
            return np.full(len(X), classes_[0] if (classes_ is not None and len(classes_) > 0) else 0)
            
        model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        predicted_idx = predicted.cpu().numpy()
        return np.array([classes_[i] for i in predicted_idx])
    
    def predict_proba(self, X):
        classes_ = getattr(self, "classes_", None)
        model = getattr(self, "model", None)
        if not PYTORCH_AVAILABLE or model is None or classes_ is None or len(classes_) == 0:
            probas = np.zeros((len(X), len(classes_) if (classes_ is not None and len(classes_) > 0) else 1))
            probas[:, 0] = 1.0
            return probas
            
        model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = model(X_tensor)
            probas = torch.softmax(outputs, dim=1).cpu().numpy()
        return probas
    
    def get_params(self, deep=True):
        return {"hidden_size": self.hidden_size, "epochs": self.epochs, "batch_size": self.batch_size, "lr": self.lr}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

class RNNClassifier(DeepLearningBaseClassifier):
    pass

def get_models_and_params():
    """
    Returns a dictionary of models and their hyperparameter grids for HP-tuning.
   
    """
    models = {
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': [
                {'n_neighbors': np.arange(1, 50, 2), 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan', 'chebyshev']},
                {'n_neighbors': np.arange(1, 50, 2), 'weights': ['uniform', 'distance'], 'metric': ['minkowski'], 'p': [3, 4]}
            ]
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': [None, 'balanced']
            }
        },
        'LinearSVC': {
            'model': LinearSVC(random_state=RANDOM_STATE, max_iter=100, dual=False),
            'params': [
                {'C': np.logspace(-4, 3, 20), 'penalty': ['l2'], 'loss': ['squared_hinge'], 'dual': [False], 'class_weight': [None, 'balanced'], 'tol': [1e-4, 1e-3], 'fit_intercept': [True, False]},
                {'C': np.logspace(-4, 3, 20), 'penalty': ['l1'], 'loss': ['squared_hinge'], 'dual': [False], 'class_weight': [None, 'balanced'], 'tol': [1e-4, 1e-3], 'fit_intercept': [True, False]}
            ]
        },
        'LogisticRegression': {
            'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=100),
            'params': [
                {'C': np.logspace(-4, 3, 20), 'penalty': ['l2'], 'solver': ['lbfgs', 'sag'], 'class_weight': [None, 'balanced'], 'fit_intercept': [True, False]},
                {'C': np.logspace(-4, 3, 20), 'penalty': ['l1', 'l2'], 'solver': ['saga'], 'class_weight': [None, 'balanced'], 'fit_intercept': [True, False]}
            ]
        },

        'RNN': {
            'model': RNNClassifier(hidden_size=64, epochs=10, batch_size=32, lr=0.001),
            'params': {
                'hidden_size': [64],
                'epochs': [10],
                'lr': [1e-3, 5e-3],
                'batch_size': [32, 64]
            }
        }
    }
    return models

def _count_param_combinations(param_dist):
    try:
        if isinstance(param_dist, list):
            return int(sum([np.prod([len(v) for v in p.values()]) for p in param_dist]))
        return int(np.prod([len(v) for v in param_dist.values()]))
    except Exception:
        return None

def tune_hyperparameters(model, param_dist, X_train, y_train, groups=None, n_iter=100, cv=3):
    """
    Perform HalvingRandomSearchCV (or full Stratified RandomizedSearchCV for small grids/sequence models)
    to find optimal hyperparameters quickly and reliably.
   
    """
    total_combinations = _count_param_combinations(param_dist)
    n_candidates = min(n_iter, total_combinations) if total_combinations is not None else n_iter

    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    n_classes = len(unique_classes)
    min_class_count = int(np.min(class_counts))

    if groups is not None:
        actual_cv = cv
        cv_splitter = StratifiedGroupKFold(n_splits=actual_cv)
    else:
        # When each class has only 1 sequence (e.g. RNN at SeedsPerLabel=1), stratified CV is impossible
        if min_class_count < 2:
            fitted_model = clone(model)
            fitted_model.fit(X_train, y_train)
            return fitted_model, fitted_model.get_params()

        actual_cv = min(cv, min_class_count)
        cv_splitter = StratifiedKFold(n_splits=actual_cv, shuffle=True, random_state=RANDOM_STATE)

        # For small parameter grids (e.g. RNN with 4 combinations) or small sequence counts,
        # evaluate on full stratified folds instead of subsampling below batch_size
        if (total_combinations is not None and total_combinations <= 15) or len(X_train) < n_classes * actual_cv * 4:
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dist,
                n_iter=n_candidates,
                cv=cv_splitter,
                scoring='accuracy',
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE,
                verbose=1
            )
            search.fit(X_train, y_train)
            return search.best_estimator_, search.best_params_

    # Ensure iter 0 has enough training samples per fold (>= 55) so KNN n_neighbors<=49 and group splits do not fail
    min_for_train = int(np.ceil(55 * actual_cv / max(1, actual_cv - 1)))
    min_res = min(len(X_train), max(n_classes * actual_cv * 2, min_for_train))

    search = HalvingRandomSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_candidates=n_candidates,
        min_resources=min_res,
        cv=cv_splitter,
        scoring='accuracy',
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbose=1
    )

    if groups is not None:
        search.fit(X_train, y_train, groups=groups)
    else:
        search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def tune_hyperparameters_custom_val(model, param_dist, X_train, y_train, X_val, y_val, n_iter=100):
    """
    Perform HalvingRandomSearchCV with a PredefinedSplit to strictly train on X_train and validate on X_val.
    """
    # Combine data
    X_combined = np.vstack((X_train, X_val))
    y_combined = np.hstack((y_train, y_val))
    
    # test_fold: -1 means train only, 0 means test (validation) only
    test_fold = np.concatenate([
        np.full(len(X_train), -1),
        np.zeros(len(X_val))
    ])
    
    ps = PredefinedSplit(test_fold)

    total_combinations = _count_param_combinations(param_dist)
    n_candidates = min(n_iter, total_combinations) if total_combinations is not None else n_iter

    n_total = len(X_combined)
    n_train = len(X_train)
    n_val = len(X_val)
    n_classes = len(np.unique(y_combined))

    # Guarantee iter 0 subsamples enough rows to always populate validation (test_fold == 0)
    # across all classes AND provide >= 65 training samples so KNN (n_neighbors <= 49) never fails
    min_for_val = int(np.ceil(max(30, n_classes * 3) * n_total / max(1, n_val)))
    min_for_train = int(np.ceil(65 * n_total / max(1, n_train)))
    min_res = min(n_total, max(n_classes * 2, min_for_val, min_for_train))
    
    # refit=False avoids a redundant full fit on (X_train + X_val); caller fits strictly on X_train
    search = HalvingRandomSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_candidates=n_candidates,
        min_resources=min_res,
        cv=ps,
        scoring='accuracy',
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        refit=False,
        verbose=1
    )
        
    search.fit(X_combined, y_combined)
    best_estimator = clone(model).set_params(**search.best_params_)
    return best_estimator, search.best_params_
