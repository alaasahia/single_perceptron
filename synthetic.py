import numpy as np
def generate_synthetic_data(class_n, n_features, center1=None, center2=None):
    if center1 is None:
        center1 = np.full(n_features, -2)
    if center2 is None:
        center2 = np.full(n_features, 2)
    
    x1 = np.random.randn(class_n, n_features)
    x1_centered = x1 + center1 
    y1 = np.zeros((class_n, 1))
    
    x2 = np.random.randn(class_n, n_features)
    x2_centered = x2 + center2 
    y2 = np.ones((class_n, 1))
    
    x = np.vstack((x1_centered, x2_centered))
    y = np.vstack((y1, y2))
    data = np.hstack((x, y))
    np.random.shuffle(data)
    
    x_shuf = data[:, :n_features]
    y_shuf = data[:, n_features]
    
    return x_shuf, y_shuf
