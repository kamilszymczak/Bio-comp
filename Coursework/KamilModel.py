import numpy as np
import inspect

def __weight_matrix__(x, y):
        """Return a numpy array of random weights with the specified size

        :param x: number of rows in the desired matrix
        :type x: int
        :param y: number of columns in the desired matrix
        :type y: int
        """
        return 2*np.random.rand(x, y)-1


def __single_layer_forward_propagation__(N_prev, W_curr, bias, activation):
    Z_curr = np.dot(N_prev, W_curr)
    Z_curr = Z_curr + bias

    # TODO: add more activation functions
    if activation is "sigmoid":
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
        self.col_int=None
        self.input=None
        self.compiled=False

    def add(self, layer):
        """Add a new layer to the Neural Network

        :param layer: Add an instance of the layer class
        :type layer: Layer
        """
        if self.input == None:
            raise Exception("Define input first before creating layers.")
        if not inspect.isclass(Layer):
            raise Exception("Parameter must be an instance of the Layer class.")
        self.layers.append(layer)


    def set_input(self, input_matrix):
        """Insert input matrix and Set the number of columns in the input matrix for the ANN

        :param input_matrix: matrix of input data
        :type col_int: int
        """
        # self.input_cols = input_matrix.shape[1]
        self.layers.append(Layer(input_matrix.shape[1], use_bias=True))
        self.input = input_matrix

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

    def epoch(self):
        """One epoch of the Neural Network

        :param input_matrix: The matrix provided to the neural network
        :type input_matrix: numpy.ndarray
        :raises Exception: Not Compiled exception
        :raises Exception: Misconfigured input columns exception
        """
        if not self.compiled:
            raise Exception("The neural network must be compiled before performing training or inference.")
        if not input_matrix.shape[1] == self.weights[0]:
            raise Exception("The input columns have been misconfigured. Expected: ", self.input_cols, "Actual: ", input_matrix.shape[1])

        # First layer / input and first set of weights (OLD)
        # self.layer_outputs = [__single_layer_forward_propagation__(self.input, self.weights[0], self.bias[0], self.layers[0].activation)]
        # for i in range(1, len(self.weights)):
        #     self.layer_outputs = self.layer_outputs + [__single_layer_forward_propagation__(self.layer_outputs[i-1], self.weights[i], self.bias[i], self.layers[i].activation)]

        # NEW
        for i in range(1, len(self.layers)):
            if i == len(self.layers)-1:
                output = __single_layer_forward_propagation__(self.layer[i-1].neurons_val, self.layer[i-1].weights, self.layer[i-1].bias, self.layers[i-1].activation)
                output_layer = Layer(output.shape[1])
                output_layer.neurons_val = output
                break

            self.layer[i].neurons_val = __single_layer_forward_propagation__(self.layer[i-1].neurons_val, self.layer[i-1].weights, self.layer[i-1].bias, self.layers[i-1].activation)

        return output_layer

    def __generate_weights__(self):
        """Generates the weight matrices & bias vectors for the ANN Layers
        """
        neuron_definition = [self.input_cols]
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

        for i in range(len(self.layers)):
            self.layers[i].weigths = __weight_matrix__(neuron_definition[i-1], neuron_definition[i])
            if self.layers[i].use_bias:
                self.layers[i].bias = 2*np.random.rand(self.layers[i].neurons)-1

class Layer:
    """Layer class used to add layers to the ANN
    """
    def __init__(self, neurons, activation=None, use_bias=True):
        """Construct a new Layer
        """
        self.neurons = neurons
        self.neurons_val = []
        self.weights = []
        self.bias = []
        self.activation = activation
        self.use_bias = use_bias

        

