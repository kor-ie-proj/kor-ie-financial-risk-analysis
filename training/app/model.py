import numpy as np
from sklearn.linear_model import LinearRegression

def train_dummy(n=500):
    rng = np.random.default_rng(0)
    X = rng.random((n, 3))   # base_rate, ccsi, leverage
    y = (X[:,1] - X[:,0]*0.2 - X[:,2]*0.1)
    return LinearRegression().fit(X, y)
