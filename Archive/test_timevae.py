import numpy as np
from augmentations import apply_timevae

def test_timevae():
    print ===")
    # Create dummy data: 30 samples, each 100 timesteps, 5 features
    n_samples = 30
    timesteps = 100
    n_features = 5
    
    X = np.random.randn(n_samples, timesteps, n_features)
    
    # Tạo nhãn giả: 2 class, mỗi class 15 mẫu
    y = np.array([0] * 15 + [1] * 15)
    
    print
    print
    print
    print)}")
    
    print ===")
    print
    
    # Chạy TimeVAE
    # scale_factor = 2 nghĩa là dữ liệu đầu ra sẽ gấp đôi dữ liệu đầu vào
    X_syn, y_syn = apply_timevae(X, y, scale_factor=2, epochs=10)
    
    print
    print
    print
    
    if X_syn.shape[0] == 60:
        print.")
    else:
        print.")

if __name__ == "__main__":
    test_timevae()
