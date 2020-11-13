# Loss functions here
import numpy as np

def mean_squared_error(y, yhat):
    return np.mean((y - yhat)**2)