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

def __calculate_one_layer__(input_matrix, weight_matrix, bias):
    """Calculate the output of a single layer

    :param input_matrix: The input matrix to the layer
    :type input_matrix: numpy.ndarray
    :param weight_matrix: The matrix of weights for the layer
    :type weight_matrix: numpy.ndarray
    :param bias: The bias vector for the layer
    :type bias: numpy.ndarray
    :raises Exception: Dot product rule violation
    :raises Exception: Bias doesnt match number of neurons in layer
    :return: Returns the output of each neuron in the layer
    :rtype: numpy.ndarray
    """
    if not (input_matrix.shape[1] == weight_matrix.shape[0]):
        raise Exception("The number of columns in the input_matrix must equal the number of rows in the weight_matrix.")
    
    if not (bias.shape[0] == weight_matrix.shape[1]):
        raise Exception("The number of elements in the bias vector should be the same as the number of columns in the weigth matrix.")
    out = np.dot(input_matrix, weight_matrix)
    out = out + bias
    return out

def __apply_activation__(weighted_sum, activation_func):
    """[summary]

    :param weighted_sum: [description]
    :type weighted_sum: [type]
    :param activation_func: [description]
    :type activation_func: [type]
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
        # List of layers
        self.layers=[]
        # List of bias
        self.bias = []
        self.input_cols=None
        self.compiled=False

    def add(self, layer):
        """Add a new layer to the Neural Network

        :param layer: Add an instance of the layer class
        :type layer: Layer
        """
        if not inspect.isclass(Layer):
            raise Exception("Parameter must be an instance of the Layer class.")
        self.layers = self.layers + [layer]

    def set_input_cols(self, col_int):
        """Set the number of columns in the input matrix for the ANN

        :param col_int: Integer of the number of rows
        :type col_int: int
        """
        self.input_cols = col_int

    def compile(self):
        """Compile the Neural Network so it is ready for training or inference
        """
        if self.input_cols == None:
            raise Exception("Must define the size of the input matrix.")
        if len(self.layers) < 2:
            raise Exception("The ANN needs at least 1 hidden and 1 output layer.")
        self.__generate_weights__()
        self.compiled = True
        print("Model Compiled!")

    def epoch(self, input_matrix):
        """One epoch of the Neural Network

        :param input_matrix: The matrix provided to the neural network
        :type input_matrix: numpy.ndarray
        :raises Exception: Not Compiled exception
        :raises Exception: Misconfigured input columns exception
        """
        if not self.compiled:
            raise Exception("The neural network must be compiled before performing training or inference.")
        if not input_matrix.shape[1] == self.weights[0].shape[0]:
            raise Exception("The input columns have been misconfigured. Expected: ", self.input_cols, "Actual: ", input_matrix.shape[1])
        # Input layer special case
        weighted_sum = __calculate_one_layer__(input_matrix, self.weights[0], self.bias[0])
        with_activation = __apply_activation__(weighted_sum, self.layers[0].activation)
        self.layer_outputs = [with_activation]
        # Hidden layers -> output layer
        for i in range(1, len(self.weights)):
            weighted_sum = __calculate_one_layer__(self.layer_outputs[i-1], self.weights[i], self.bias[i])
            with_activation = __apply_activation__(weighted_sum, self.layers[i].activation)
            self.layer_outputs = self.layer_outputs + [with_activation]

    def __generate_weights__(self):
        """Generates the weight matrices & bias vectors for the ANN Layers
        """
        num_cols = self.input_cols
        neuron_definition = [num_cols]
        temp_bias = []
        
        # Build the bias list & collect the number of neurons into a single list
        for i in range(len(self.layers)):
            temp_bias = temp_bias + [2*np.random.rand(self.layers[i].neurons)-1]
            neuron_definition = neuron_definition + [self.layers[i].neurons]
        
        # construct the weight matrices
        # neuron_definition[0] is the number of columns of the input layer
        temp_weights = []
        for i  in range(1, len(neuron_definition)):
            temp_weights = temp_weights + [__weight_matrix__(neuron_definition[i-1], neuron_definition[i])]
        
        self.weights = temp_weights
        self.bias = temp_bias

class Layer:
    """Layer class used to add layers to the ANN
    """
    def __init__(self, neurons, activation="null", use_bias=True):
        self.neurons = neurons
        self.activation = __enumerate_activation__(activation)
        self.use_bias = use_bias
        

