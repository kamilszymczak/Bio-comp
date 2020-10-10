import numpy as np
import inspect
import enum

ActivationFunction = enum.Enum('ActivationFunction', 'NULL SIGMOID HYPERBOLIC_TANGENT COSINE GAUSSIAN RELU')

def __weight_matrix__(x, y):
        """Return a numpy array of random weights with the specified size

        :param x: number of rows in the desired matrix
        :type x: int
        :param y: number of columns in the desired matrix
        :type y: int
        """
        return 2*np.random.rand(x, y)-1

def __calculate_one_layer__(input_matrix, layer):
    """Calculate the output of a single layer

    :param input_matrix: The input matrix to the layer
    :type input_matrix: numpy.ndarray
    :param layer: The layer being calculated
    :type layer: Layer
    :raises Exception: Dot product rule violation
    :raises Exception: Bias doesnt match number of neurons in layer
    :return: Returns the output of each neuron in the layer
    :rtype: numpy.ndarray
    """
    if not (input_matrix.shape[1] == layer.weights.shape[0]):
        raise Exception("The number of columns in the input_matrix must equal the number of rows in the weight_matrix.")
    
    if not (layer.bias.shape[0] == layer.weights.shape[1]):
        raise Exception("The number of elements in the bias vector should be the same as the number of columns in the weigth matrix.")
    out = np.dot(input_matrix, layer.weights)
    if layer.use_bias:
        out = out + layer.bias
    return __apply_activation__(out, layer.activation)

def __apply_activation__(weighted_sum, activation_func):
    """Apply activation function to matrix

    :param weighted_sum: The matrix of the sum of input * weight + bias
    :type weighted_sum: numpy.ndarray
    :param activation_func: The activation function identifier
    :type activation_func: Enum.ActivationFunction
    """
    af = __pick_activation__(activation_func)
    if af == None:
        raise Exception("Invalid activation function")
    return af(weighted_sum)

# TODO: IMPLEMENT FUNCTION
def __null__(z):
    return z

def __sigmoid__(z):
    """Applies sigmoid activation function

    :param z: The matrix of weighted sum values for each neuron
    :type z: numpy.ndarray
    :return: The matrix with sigmoid applied to each element
    :rtype: numpy.ndarray
    """
    return 1.0/(1.0 +np.exp(-z))

# TODO: IMPLEMENT FUNCTION
def __hyperbolic__tangent__(z):
    return z

# TODO: IMPLEMENT FUNCTION
def __cosine__(z):
    return z

# TODO: IMPLEMENT FUNCTION
def __gaussian__(z):
    return z

def __relu__(z):
    """Applies relu activation function

    :param z: The matrix of weighted sum values for each neuron
    :type z: numpy.ndarray
    :return: The matrix with relu applied to each element
    :rtype: numpy.ndarray
    """
    return z if z > 0 else 0
    
def __pick_activation__(activation):
    """Select activation function to use

    :param activation: The activation function identifier
    :type activation: Enum.ActivationFunction
    :return: reference to the activation function defintion
    :rtype: fn
    """
    activation_picker = {
        ActivationFunction.NULL: __null__,
        ActivationFunction.SIGMOID: __sigmoid__,
        ActivationFunction.HYPERBOLIC_TANGENT: __hyperbolic__tangent__,
        ActivationFunction.COSINE: __cosine__,
        ActivationFunction.GAUSSIAN: __gaussian__,
        ActivationFunction.RELU: __relu__
    }
    return activation_picker.get(activation)

def __enumerate_activation__(activation_string):
    """Enumerate the activation function

    :param activation_string: string representing the activation function
    :type activation_string: string
    :return: The enumerated activation function
    :rtype: Enum.ActivationFunction
    """
    activation_string = activation_string.lower()
    activation_enum = {
        "null": ActivationFunction.NULL,
        "sigmoid": ActivationFunction.SIGMOID,
        "hyperbolictangent": ActivationFunction.HYPERBOLIC_TANGENT,
        "tan": ActivationFunction.HYPERBOLIC_TANGENT,
        "cosine": ActivationFunction.COSINE,
        "gaussian": ActivationFunction.GAUSSIAN,
        "relu": ActivationFunction.RELU
    }
    return activation_enum.get(activation_string, ActivationFunction.NULL)

class ANN:
    """Artificial Neural Network class Implementation
    """
    def __init__(self):
        """Construct a new sequential Artificial Neural Network
        """
        self.layers=[]
        self.input=None
        self.input_column_size=None
        self.compiled=False

    def add(self, layer):
        """Add a new layer to the Neural Network

        :param layer: Add an instance of the layer class
        :type layer: Layer
        """
        if not inspect.isclass(Layer):
            raise Exception("Parameter must be an instance of the Layer class.")
        self.layers = self.layers + [layer]
        self.compiled = False

    def set_input(self, input_matrix, result_vector):
        """Provide the input data to the ANN

        :param input_matrix: The input matrix
        :type input_matrix: np.ndarray
        :param result_vector: A vector or results. Specifies the 'Y' values to calculate the loss when training
        :type result_vector: numpy.ndarray
        :raises Exception: If input matrix columns doesnt match already configured columns
        """
        if input_matrix.shape[0] != result_vector.shape[0]:
            raise Exception("Number of rows in the input matrix and result vector are not compatible")
        if self.input_column_size is None:
            self.input_column_size = input_matrix.shape[1]
            self.input = input_matrix
            self.compiled = False

        elif self.input_column_size == input_matrix.shape[1]:
            self.input = input_matrix
        else:
            raise Exception("Weights might have been generated for a different shape input, please check the columns of your input")
    
    def compile(self):
        """Compile the Neural Network so it is ready for training or inference
        """
        if self.input is None:
            raise Exception("Must define the size of the input matrix.")
        if len(self.layers) < 1:
            raise Exception("The ANN needs at least 1 layer.")
        self.__generate_weights__()
        self.compiled = True
        print("Model Compiled!")

    def one_pass(self):
        """One pass of data in input through the Neural Network

        :raises Exception: Not Compiled exception
        """
        if not self.compiled:
            raise Exception("The neural network must be compiled before performing training or inference.")
        # Input layer special case
        self.layers[0].output = __calculate_one_layer__(self.input, self.layers[0])
        # Hidden layers -> output layer
        for i in range(1, len(self.layers)):
            self.layers[i].output = __calculate_one_layer__(self.layers[i-1].output, self.layers[i])

    def __generate_weights__(self):
        """Generates the weight matrices & bias vectors for the ANN Layers
        """

        # raise exception if no input
        
        # special case for weights & bias from input layer
        self.layers[0].bias = 2*np.random.rand(self.layers[0].neurons)-1
        self.layers[0].weights = __weight_matrix__(self.input.shape[1], self.layers[0].neurons)
        
        # Construct weight matrices & bias of each layer
        for i in range(1, len(self.layers)):
            self.layers[i].bias = 2*np.random.rand(self.layers[i].neurons)-1
            self.layers[i].weights = __weight_matrix__(self.layers[i-1].neurons, self.layers[i].neurons)
        

class Layer:
    """Layer class used to add layers to the ANN
    """
    def __init__(self, neurons, activation="null", use_bias=True):
        """Constuctor for Layer class

        :param neurons: The number of neurons in the layer
        :type neurons: int
        :param activation: The activation funciton for the layer, defaults to "null"
        :type activation: str, optional
        :param use_bias: If you use a bias, defaults to True
        :type use_bias: bool, optional
        """
        self.neurons = neurons
        self.activation = __enumerate_activation__(activation)
        self.use_bias = use_bias
        self.weights = None
        self.bias = None
        self.output = None
        

