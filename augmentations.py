# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from scipy.interpolate import CubicSpline
# pyrefly: ignore [missing-import]
from imblearn.over_sampling import SMOTE, ADASYN

# -------------------------------------------------------------
# 6 Synchronous Signal Augmentations
# -------------------------------------------------------------

def jitter(x, sigma=0.03):
    """
    Adds random Gaussian noise (0 to 0.05) to every data point.
   
    """
    noise = np.random.normal(loc=0, scale=sigma, size=x.shape)
    return x + noise

def scaling(x, sigma=0.1):
    """
    Multiplies the entire signal sequence by a random scalar.
   
    """
    if len(x.shape) == 3:
        factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], x.shape[2]))
        return np.multiply(x, factor[:, np.newaxis, :])
    else:
        factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], 1))
        return np.multiply(x, factor)

# def permutation(x, nPerm=4, minSegLength=10):
#     """
#     Divides the signal into equal segments and randomly shuffles their order.
#    
#     """
#     if len(x.shape) != 3:
#         print("Warning: permutation augmentation expects 3D shape, returning original")
#         return x
#         
#     max_possible_seg_len = x.shape[1] // nPerm
#     if max_possible_seg_len < 1:
#         print("Warning: time series too short for permutation, returning original")
#         return x
#     
#     effective_min_seg = min(minSegLength, max_possible_seg_len)
#     
#     x_new = np.zeros(x.shape)
#     idx = np.random.permutation(nPerm)
#     bWhile = True
#     
#     attempts = 0
#     while bWhile == True and attempts < 1000:
#         segs = np.zeros(nPerm+1, dtype=int)
#         
#         low_bound = effective_min_seg
#         high_bound = x.shape[1] - effective_min_seg
#         
#         if low_bound >= high_bound:
#             segs[1:-1] = np.linspace(0, x.shape[1], nPerm+1)[1:-1].astype(int)
#             bWhile = False
#             break
#             
#         segs[1:-1] = np.sort(np.random.randint(low_bound, high_bound, nPerm-1))
#         segs[-1] = x.shape[1]
#         
#         if np.min(np.diff(segs)) >= effective_min_seg:
#             bWhile = False
#         attempts += 1
#         
#     if attempts >= 1000:
#         segs[1:-1] = np.linspace(0, x.shape[1], nPerm+1)[1:-1].astype(int)
#     pp = 0
#     for ii in range(nPerm):
#         x_temp = x[:, segs[idx[ii]]:segs[idx[ii]+1], :]
#         x_new[:, pp:pp+len(x_temp[0]), :] = x_temp
#         pp += len(x_temp[0])
#     return x_new

def magnitude_warping(x, sigma=0.2, knot=4):
    """
    Multiplies the signal by a smooth curve generated via Cubic Spline interpolation.
   
    """
    if len(x.shape) != 3:
        print("Warning: magnitude_warping expects 3D shape, returning original")
        return x
        
    orig_steps = np.arange(x.shape[1])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper
    return ret

def time_warping(x, sigma=0.2, knot=4):
    """
    Distorts the intervals between time steps using a Cubic Spline curve.
   
    """
    if len(x.shape) != 3:
        print("Warning: time_warping expects 3D shape, returning original")
        return x
        
    orig_steps = np.arange(x.shape[1])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])(orig_steps)
            scale = (x.shape[1]-1) / time_warp[-1]
            ret[i, :, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[1]-1), pat[:, dim])
    return ret

# def window_slicing(x, reduce_ratio=0.9):
#     """
#     Extracts a random continuous sub-segment and interpolates it back to the original size.
#    
#     """
#     if len(x.shape) != 3:
#         print("Warning: window_slicing expects 3D shape, returning original")
#         return x
#         
#     target_len = np.ceil(reduce_ratio * x.shape[1]).astype(int)
#     if target_len >= x.shape[1]:
#         return x
#     starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
#     ends = (target_len + starts).astype(int)
#     
#     ret = np.zeros_like(x)
#     for i, pat in enumerate(x):
#         for dim in range(x.shape[2]):
#             ret[i, :, dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i], dim])
#     return ret

# -------------------------------------------------------------
# Mixup Strategy
# -------------------------------------------------------------
def mixup(x, y, alpha=0.2):
    """
    Performs a weighted linear combination of two random samples belonging to the same class.
   
    """
    x_new = np.zeros_like(x)
    y_new = np.copy(y)
    
    for i in range(len(x)):
        same_class_idx = np.where(y == y[i])[0]
        if len(same_class_idx) > 1:
            j = np.random.choice(same_class_idx)
            lam = np.random.beta(alpha, alpha)
            x_new[i] = lam * x[i] + (1 - lam) * x[j]
        else:
            x_new[i] = x[i]
            
    return x_new, y_new

# -------------------------------------------------------------
# Distance-Based Interpolations
# -------------------------------------------------------------
def apply_smote(X, y, scale_factor=2):
    """
    Synthesizes new examples along the line segments joining k-nearest neighbors.
   
    """
    from collections import Counter
    
    counts = Counter(y)
    sampling_strategy = {c: count * scale_factor for c, count in counts.items()}
    
    original_shape = X.shape
    if len(original_shape) == 3:
        X_flat = X.reshape(original_shape[0], -1)
    else:
        X_flat = X
        
    min_count = min(counts.values())
    if min_count <= 1:
        return X, y
        
    k_neighbors = min(5, min_count - 1)
    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_flat, y)
    
    if len(original_shape) == 3:
        X_resampled = X_resampled.reshape(X_resampled.shape[0], original_shape[1], original_shape[2])
        
    return X_resampled, y_resampled

def apply_adasyn(X, y, scale_factor=2):
    """
    Similar to SMOTE, but dynamically focuses on hard-to-learn examples.
   
    """
    from collections import Counter
    
    counts = Counter(y)
    sampling_strategy = {c: count * scale_factor for c, count in counts.items()}
    
    original_shape = X.shape
    if len(original_shape) == 3:
        X_flat = X.reshape(original_shape[0], -1)
    else:
        X_flat = X
        
    min_count = min(counts.values())
    if min_count <= 1:
        return X, y
        
    k_neighbors = min(5, min_count - 1)
    
    # Do not catch or mask errors. Let it fail so the benchmark accurately 
    # reflects ADASYN's inability to handle perfectly separated classes.
    #
    adasyn = ADASYN(sampling_strategy=sampling_strategy, n_neighbors=k_neighbors, random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X_flat, y)
    
    if len(original_shape) == 3:
        X_resampled = X_resampled.reshape(X_resampled.shape[0], original_shape[1], original_shape[2])
        
    return X_resampled, y_resampled

# -------------------------------------------------------------
# Generative Adversarial Networks (GANs) / VAEs
# -------------------------------------------------------------
_timevae_cache = {}

def apply_timevae(X, y, scale_factor=2, epochs=50, batch_size=32, latent_dim=16):
    """
    Generates synthetic data using a PyTorch Variational Autoencoder (TimeVAE).
   
    """
    global _timevae_cache
    
    try:
        # pyrefly: ignore [missing-import]
        import torch
        # pyrefly: ignore [missing-import]
        import torch.nn as nn
        # pyrefly: ignore [missing-import]
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        from config import USE_GPU

        PYTORCH_AVAILABLE = True
    except ImportError:
        print("PyTorch is not installed. Cannot run TimeVAE.")
        return X, y
        
    original_shape = X.shape
    if len(original_shape) != 3:
        print("Warning: TimeVAE requires 3D shape, returning original")
        return X, y
        
    timesteps = original_shape[1]
    features = original_shape[2]
    
    if USE_GPU:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        
    class TimeVAE(nn.Module):
        def __init__(self, seq_len, n_features, latent_dim):
            super(TimeVAE, self).__init__()
            self.seq_len = seq_len
            self.n_features = n_features
            self.latent_dim = latent_dim
            
            self.encoder_gru = nn.GRU(n_features, 64, batch_first=True)
            self.fc_mu = nn.Linear(64, latent_dim)
            self.fc_logvar = nn.Linear(64, latent_dim)
            
            self.fc_dec = nn.Linear(latent_dim, 64)
            self.decoder_gru = nn.GRU(64, 64, batch_first=True)
            self.fc_out = nn.Linear(64, n_features)
            
        def encode(self, x):
            _, h_n = self.encoder_gru(x)
            h = h_n[-1]
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar
            
        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
            
        def decode(self, z):
            h = self.fc_dec(z)
            h_seq = h.unsqueeze(1).repeat(1, self.seq_len, 1)
            out, _ = self.decoder_gru(h_seq)
            out = self.fc_out(out)
            return out
            
        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)
            return x_recon, mu, logvar

    def loss_function(recon_x, x, mu, logvar):
        BCE = nn.MSELoss(reduction='sum')(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    X_syn_list = [X]
    y_syn_list = [y]
    
    cache_key = (X.shape, np.round(np.sum(X), 4))
    if cache_key not in _timevae_cache:
        _timevae_cache[cache_key] = {}
    
    classes = np.unique(y)
    for c in classes:
        X_c = X[y == c]
        if len(X_c) == 0:
            continue
            
        n_samples_to_generate = len(X_c) * (scale_factor - 1)
        if n_samples_to_generate <= 0:
            continue
            
        if c in _timevae_cache[cache_key]:
            model = _timevae_cache[cache_key][c]
        else:
            X_tensor = torch.FloatTensor(X_c)
            dataset = TensorDataset(X_tensor)
            
            current_batch_size = min(batch_size, len(X_c))
            if current_batch_size < 1:
                continue
                
            dataloader = DataLoader(dataset, batch_size=current_batch_size, shuffle=True)
            
            model = TimeVAE(seq_len=timesteps, n_features=features, latent_dim=latent_dim).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            model.train()
            for epoch in range(epochs):
                for batch_x in dataloader:
                    batch_x = batch_x[0].to(device)
                    optimizer.zero_grad()
                    recon_batch, mu, logvar = model(batch_x)
                    loss = loss_function(recon_batch, batch_x, mu, logvar)
                    loss.backward()
                    optimizer.step()
            
            _timevae_cache[cache_key][c] = model
                
        model.eval()
        with torch.no_grad():
            z = torch.randn(n_samples_to_generate, latent_dim).to(device)
            generated_samples = model.decode(z).cpu().numpy()
            
        X_syn_list.append(generated_samples)
        y_syn_list.append(np.full(n_samples_to_generate, c))
        
    if len(X_syn_list) > 1:
        return np.vstack(X_syn_list), np.hstack(y_syn_list)
    return X, y

# -------------------------------------------------------------
# Probabilistic Methods (GMM / HMM-GMM)
# -------------------------------------------------------------
def apply_gmm(X, y, scale_factor=2):
    """
    Generates synthetic data using Gaussian Mixture Models.
   
    """
    from sklearn.mixture import GaussianMixture
    
    original_shape = X.shape
    if len(original_shape) == 3:
        X_flat = X.reshape(original_shape[0], -1)
    else:
        X_flat = X
        
    X_syn_list = [X]
    y_syn_list = [y]
    
    classes = np.unique(y)
    for c in classes:
        X_c = X_flat[y == c]
        if len(X_c) == 0:
            continue
            
        n_samples_to_generate = len(X_c) * (scale_factor - 1)
        if n_samples_to_generate <= 0:
            continue
            
        n_components = min(max(1, len(X_c) // 2), 5)
        gmm = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=42)
        
        try:
            # Add tiny noise to prevent degenerate mixture covariance
            #
            X_c_fit = X_c + np.random.normal(0, 1e-4, size=X_c.shape)
            gmm.fit(X_c_fit)
            X_syn_c, _ = gmm.sample(n_samples_to_generate)
            
            if len(original_shape) == 3:
                X_syn_c = X_syn_c.reshape(n_samples_to_generate, original_shape[1], original_shape[2])
                
            X_syn_list.append(X_syn_c)
            y_syn_list.append(np.full(n_samples_to_generate, c))
        except Exception as e:
            print(f"GMM fitting failed for class {c}: {e}")
            
    if len(X_syn_list) > 1:
        return np.vstack(X_syn_list), np.hstack(y_syn_list)
    return X, y

def apply_hmm_gmm(X, y, scale_factor=2):
    """
    Generates synthetic data using HMM-GMM (hmmlearn).
   
    """
    try:
        # pyrefly: ignore [missing-import]
        from hmmlearn import hmm
    except ImportError:
        print("hmmlearn not installed. Cannot run HMM-GMM.")
        return X, y
        
    X_syn_list = [X]
    y_syn_list = [y]
    
    classes = np.unique(y)
    for c in classes:
        X_c = X[y == c] if len(X.shape) == 3 else X[y == c].reshape(-1, 1, X.shape[1])
        if len(X_c) == 0:
            continue
            
        n_samples_to_generate = len(X_c) * (scale_factor - 1)
        if n_samples_to_generate <= 0:
            continue
            
        lengths = [X_c.shape[1]] * len(X_c)
        X_c_flat = X_c.reshape(-1, X_c.shape[2])
        
        # Add tiny noise to prevent degenerate mixture covariance (zero variance)
        #
        X_c_flat_fit = X_c_flat + np.random.normal(0, 1e-4, size=X_c_flat.shape)
        
        n_components = max(1, min(2, len(X_c) // 2))
        n_mix = 1 
        
        # Adding Smoothings/Pseudocounts to prevent empty transition matrices on small datasets
        model = hmm.GMMHMM(
            n_components=n_components, 
            n_mix=n_mix, 
            covariance_type='diag', 
            min_covar=1e-2, 
            transmat_prior=1.1,
            startprob_prior=1.1,
            weights_prior=1.1,
            random_state=42, 
            n_iter=100
        )
        
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_c_flat_fit, lengths)
                
                generated_samples = []
                for _ in range(n_samples_to_generate):
                    samp, _ = model.sample(X_c.shape[1])
                    generated_samples.append(samp)
                    
                X_syn_c = np.array(generated_samples)
                if len(X.shape) != 3:
                    X_syn_c = X_syn_c.reshape(n_samples_to_generate, -1)
                    
                X_syn_list.append(X_syn_c)
                y_syn_list.append(np.full(n_samples_to_generate, c))
        except Exception as e:
            print(f"HMM-GMM fitting failed for class {c}: {e}")
            
    if len(X_syn_list) > 1:
        return np.vstack(X_syn_list), np.hstack(y_syn_list)
    return X, y
