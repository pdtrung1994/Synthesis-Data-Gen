# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedGroupKFold
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

class DeepLearningBaseClassifier:
    """
    PyTorch RNN Classifier wrapper acting like a scikit-learn estimator
   
    """
    def __init__(self, hidden_size=64, epochs=10, batch_size=32, lr=0.001):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.classes_ = None
        self.model = None
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if not PYTORCH_AVAILABLE:
            print("PyTorch not installed. Skipping training.")
            return self
            
        self.class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([self.class_to_idx[c] for c in y])
        
        if USE_GPU:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
            
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
        if not PYTORCH_AVAILABLE or self.model is None or len(self.classes_) == 0:
            return np.full(len(X), self.classes_[0] if self.classes_ is not None else 0)
            
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        predicted_idx = predicted.cpu().numpy()
        return np.array([self.classes_[i] for i in predicted_idx])
    
    def predict_proba(self, X):
        if not PYTORCH_AVAILABLE or self.model is None or len(self.classes_) == 0:
            probas = np.zeros((len(X), len(self.classes_) if self.classes_ is not None else 1))
            probas[:, 0] = 1.0
            return probas
            
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
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
            'params': {
                'n_neighbors': np.arange(3, 20),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        'LinearSVC': {
            'model': LinearSVC(random_state=RANDOM_STATE, max_iter=100, dual=False),
            'params': [
                {'C': np.logspace(-3, 2, 6), 'penalty': ['l2'], 'loss': ['squared_hinge'], 'dual': [True, False]},
                {'C': np.logspace(-3, 2, 6), 'penalty': ['l2'], 'loss': ['hinge'], 'dual': [True]},
                {'C': np.logspace(-3, 2, 6), 'penalty': ['l1'], 'loss': ['squared_hinge'], 'dual': [False]}
            ]
        },
        'LogisticRegression': {
            'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=100),
            'params': [
                {'C': np.logspace(-3, 2, 6), 'penalty': ['l2', None], 'solver': ['lbfgs']}
            ]
        },

        'RNN': {
            'model': RNNClassifier(),
            'params': {
                'hidden_size': [32, 64, 128],
                'epochs': [20, 50],
                'lr': [1e-3, 1e-4]
            }
        }
    }
    return models

def tune_hyperparameters(model, param_dist, X_train, y_train, groups=None, n_iter=100, cv=3):
    """
    Perform HalvingRandomSearchCV to find optimal hyperparameters quickly.
   
    """
    if groups is not None:
        cv_splitter = StratifiedGroupKFold(n_splits=cv)
    else:
        cv_splitter = cv
        
    search = HalvingRandomSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_candidates=n_iter,
        cv=cv_splitter,
        scoring='accuracy',
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbose=1
    )
    
    # Catching edge case where parameter space is smaller than n_candidates
    #
    import itertools
    try:
        if isinstance(param_dist, list):
            total_combinations = sum([np.prod([len(v) for v in p.values()]) for p in param_dist])
        else:
            total_combinations = np.prod([len(v) for v in param_dist.values()])
            
        if total_combinations < n_iter:
            search.n_candidates = int(total_combinations)
    except:
        pass # In case of continuous distributions
        
    if groups is not None:
        search.fit(X_train, y_train, groups=groups)
    else:
        search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def tune_hyperparameters_custom_val(model, param_dist, X_train, y_train, X_val, y_val, n_iter=100):
    """
    Perform HalvingRandomSearchCV with a PredefinedSplit to strictly train on X_train and validate on X_val.
    """
    from sklearn.model_selection import PredefinedSplit
    
    # Combine data
    X_combined = np.vstack((X_train, X_val))
    y_combined = np.hstack((y_train, y_val))
    
    # test_fold: -1 means train only, 0 means test (validation) only
    test_fold = np.concatenate([
        np.full(len(X_train), -1),
        np.zeros(len(X_val))
    ])
    
    ps = PredefinedSplit(test_fold)
    
    search = HalvingRandomSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_candidates=n_iter,
        cv=ps,
        scoring='accuracy',
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbose=1
    )
    
    try:
        if isinstance(param_dist, list):
            total_combinations = sum([np.prod([len(v) for v in p.values()]) for p in param_dist])
        else:
            total_combinations = np.prod([len(v) for v in param_dist.values()])
            
        if total_combinations < n_iter:
            search.n_candidates = int(total_combinations)
    except:
        pass
        
    search.fit(X_combined, y_combined)
    return search.best_estimator_, search.best_params_
