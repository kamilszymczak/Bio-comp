import numpy as np

# TODO: IMPLEMENT FUNCTION
def null(z):
    return z


def sigmoid(z):
    """Applies sigmoid activation function

    :param z: The matrix of weighted sum values for each neuron
    :type z: numpy.ndarray
    :return: The matrix with sigmoid applied to each element
    :rtype: numpy.ndarray
    """
    return 1.0/(1.0 + np.exp(-z))

# TODO: IMPLEMENT FUNCTION


def hyperbolic_tangent(z):
    return z

# TODO: IMPLEMENT FUNCTION


def cosine(z):
    return z

# TODO: IMPLEMENT FUNCTION


def gaussian(z):
    return z


def relu(z):
    """Applies relu activation function

    :param z: The matrix of weighted sum values for each neuron
    :type z: numpy.ndarray
    :return: The matrix with relu applied to each element
    :rtype: numpy.ndarray
    """
    return np.where(z > 0, z, 0)

    
