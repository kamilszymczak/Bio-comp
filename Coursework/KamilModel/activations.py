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



def hyperbolic_tangent(z):
    """Applies hyperbolic tangent activation function

    :param z: The matrix of weighted sum values for each neuron
    :type z: numpy.ndarray
    :return: The matrix with hyperbolic tangent applied to each element
    :rtype: numpy.ndarray
    """
    return np.tanh(z)


def cosine(z):
    """Applies cosine activation function

    :param z: The matrix of weighted sum values for each neuron
    :type z: numpy.ndarray
    :return: The matrix with cosine applied to each element
    :rtype: numpy.ndarray
    """    
    return np.cos(z)


def gaussian(z):
    """Applies gaussian activation function

    :param z: The matrix of weighted sum values for each neuron
    :type z: numpy.ndarray
    :return: The matrix with gaussian applied to each element
    :rtype: numpy.ndarray
    """
    return (e)**((-z)**2)


def relu(z):
    """Applies relu activation function

    :param z: The matrix of weighted sum values for each neuron
    :type z: numpy.ndarray
    :return: The matrix with relu applied to each element
    :rtype: numpy.ndarray
    """
    return np.where(z > 0, z, 0)

    
