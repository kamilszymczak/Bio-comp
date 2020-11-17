# Loss functions here
import numpy as np

def mean_squared_error(y, yhat):
    return np.mean((y - yhat)**2)

# https://towardsdatascience.com/what-is-loss-function-1e2605aeb904
def hinge(y, yhat):
    return np.max([0., 1, - yhat * y])

def exponential_loss(y, yhat):
    return np.mean(np.exp(- yhat * y))