import numpy as np
import inspect
import activations


def __weight_matrix__(x, y):
        """Return a numpy array of random weights with the specified size

        :param x: number of rows in the desired matrix
        :type x: int
        :param y: number of columns in the desired matrix
        :type y: int
        """
        return 2*np.random.rand(x, y)-1


def __single_layer_forward_propagation__(N_prev, W_curr, bias, activation):
    """Return a numpy array of layer's output

    :param N_prev: matrix of previous layer's output
    :type N_prev: numpy.ndarray
    :param W_curr: matrix of weights of current layer
    :type y: numpy.ndarray
    :param bias: matrix of previous layer's output
    :type bias: numpy.ndarray
    :param activation: activation function to use
    :type activation: string
    """

    if not (N_prev.shape[1] == W_curr.shape[0]):
        raise Exception("The number of columns in the input_matrix must equal the number of rows in the weight_matrix.")
    
    if not (bias.shape[0] == W_curr.shape[1]):
        raise Exception("The number of elements in the bias vector should be the same as the number of columns in the weigth matrix.")

    Z_curr = np.dot(N_prev, W_curr)
    Z_curr = Z_curr + bias

    # TODO: add more activation functions
    if activation == "sigmoid":
        activation_fun = sigmoid

    return activation_fun(Z_curr)


class ANN:
    """Artificial Neural Network class Implementation
    """
    def __init__(self):
        """Construct a new sequential Artificial Neural Network
        """
        # List of layers
        self.layers=[]
        self.compiled=False

    def add(self, layer):
        """Add a new layer to the Neural Network

        :param layer: Add an instance of the layer class
        :type layer: Layer
        """
        if len(self.layers) < 1:
            raise Exception("Define input first before creating layers.")
        if not inspect.isclass(Layer):
            raise Exception("Parameter must be an instance of the Layer class.")
        self.layers.append(layer)


    def set_input(self, input_matrix, use_bias=True):
        """Insert input matrix and Set the number of columns in the input matrix for the ANN

        :param input_matrix: matrix of input data
        :type imput_matrix: numpy.ndarray
        :param use_bias: If to use bias or not
        :type use_bias: boolean
        """
        self.layers.append(Layer(input_matrix.shape[1], use_bias))
        self.layers[0].neurons_val = input_matrix

    def compile(self):
        """Compile the Neural Network so it is ready for training or inference
        """
        if len(self.layers) < 3:
            raise Exception("The ANN needs at least 1 hidden and 1 output layer.")
        self.__generate_weights__()
        self.compiled = True
        print("Model Compiled!")

    def epoch(self):
        """One epoch of the Neural Network

        :param input_matrix: The matrix provided to the neural network
        :type input_matrix: numpy.ndarray
        :raises Exception: Not Compiled exception
        :raises Exception: Misconfigured input columns exception
        """
        # TODO: if we want super good error checking set compiled to false every time adding new layer e.g. adding layer after compiling
        if not self.compiled:
            raise Exception("The neural network must be compiled before performing training or inference.")


        for i in range(1, len(self.layers)):
            self.layers[i].neurons_val = __single_layer_forward_propagation__(self.layers[i-1].neurons_val, self.layers[i].weights, self.layers[i-1].bias, self.layers[i].activation)

        return self.layers[-1]


    def __generate_weights__(self):
        """Generates the weight matrices & bias vectors for the ANN Layers
        """
        #Build bias for every layer except output
        for i in range(len(self.layers)-1):
            if self.layers[i].use_bias:
                self.layers[i].bias = 2*np.random.rand(self.layers[i+1].neurons)-1

        #Build weight matrix for every hidden layer
        for i in range(1, len(self.layers)):
            self.layers[i].weights = __weight_matrix__(self.layers[i-1].neurons, self.layers[i].neurons)

        # #TODO: Below should do both above in a single loop (but less readable)
        # #Build bias for every layer except output
        # for i in range(len(self.layers)-1):
        #     if self.layers[i].use_bias:
        #         self.layers[i].bias = [2*np.random.rand(self.layers[i].neurons)-1]
            
        #     self.layers[i+1].weights = __weight_matrix__(self.layers[i].neurons, self.layers[i+1].neurons)


class Layer:
    """Layer class used to add layers to the ANN
    """
    def __init__(self, neurons, activation=None, use_bias=True):
        """Construct a new Layer

        :param neurons: Number of neurons the layer will contain
        :type neurons: int
        :param activation: Activation function the layer will use, if any
        :type neurons: string
        :param use_bias: If to use bias or not
        :type use_bias: boolean
        """
        self.neurons = neurons
        self.neurons_val = []
        self.weights = None
        self.bias = None
        self.activation = activation
        self.use_bias = use_bias

        

