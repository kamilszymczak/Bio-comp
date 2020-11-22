import numpy as np
import warnings

np.seterr(divide='raise', over='warn', under='warn', invalid='ignore')

def null(z):
    """Applies no activation function

    :param z: The matrix of weighted sum values for each neuron
    :type z: numpy.ndarray
    :return: The matrix of weighted sum values for each neuron
    :rtype: numpy.ndarray
    """
    return z


def sigmoid(z):
    """Applies sigmoid activation function

    :param z: The matrix of weighted sum values for each neuron
    :type z: numpy.ndarray
    :return: The matrix with sigmoid applied to each element
    :rtype: numpy.ndarray
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.cos(z)


def gaussian(z):
    """Applies gaussian activation function

    :param z: The matrix of weighted sum values for each neuron
    :type z: numpy.ndarray
    :return: The matrix with gaussian applied to each element
    :rtype: numpy.ndarray
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.exp(-((z**2)/2))


def relu(z):
    """Applies relu activation function

    :param z: The matrix of weighted sum values for each neuron
    :type z: numpy.ndarray
    :return: The matrix with relu applied to each element
    :rtype: numpy.ndarray
    """
    #print('relu: ',z)
    return np.where(z > 0, z, 0)

def softmax(z):
    """Applies softmax activation function

    :param z: The matrix of weighted sum values for each neuron
    :type z: numpy.ndarray
    :return: The matrix with softmax applied to each element
    :rtype: numpy.ndarray
    """
    e_x = np.exp(z - np.max(z))
    return e_x / e_x.sum()   
