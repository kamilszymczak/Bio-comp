import numpy as np
import inspect
import itertools
from enum import IntEnum
from tqdm.autonotebook import tqdm
from . import activations as activ
from . import loss
from ..PSO.interface import Optimisable



class ActivationFunction(IntEnum):
    NULL = 0
    SIGMOID = 1
    HYPERBOLIC_TANGENT = 2
    COSINE = 3
    GAUSSIAN = 4
    RELU = 5
    SOFTMAX = 6

activation_picker = {
    ActivationFunction.NULL: activ.null,
    ActivationFunction.SIGMOID: activ.sigmoid,
    ActivationFunction.HYPERBOLIC_TANGENT: activ.hyperbolic_tangent,
    ActivationFunction.COSINE: activ.cosine,
    ActivationFunction.GAUSSIAN: activ.gaussian,
    ActivationFunction.RELU: activ.relu,
    ActivationFunction.SOFTMAX: activ.softmax,
}

activation_enum = {
    "null": ActivationFunction.NULL,
    "sigmoid": ActivationFunction.SIGMOID,
    "hyperbolictangent": ActivationFunction.HYPERBOLIC_TANGENT,
    "tan": ActivationFunction.HYPERBOLIC_TANGENT,
    "cosine": ActivationFunction.COSINE,
    "gaussian": ActivationFunction.GAUSSIAN,
    "relu": ActivationFunction.RELU,
    "softmax": ActivationFunction.SOFTMAX
}

loss_picker = {
    'mse': loss.mean_squared_error,
    'meansquarederror': loss.mean_squared_error,
    'hinge': loss.hinge,
    'exponentialloss': loss.exponential_loss,
    'crossentropy': loss.cross_entropy
}

def weight_matrix(x, y):
        """Return a numpy array of random weights with the specified size

        :param x: number of rows in the desired matrix
        :type x: int
        :param y: number of columns in the desired matrix
        :type y: int
        """
        return 2*np.random.rand(x, y)-1

def calculate_one_layer(input_matrix, layer):
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

    #print(out)
    #return out
    out = apply_activation(out, layer.activation)
    
    return out

def apply_activation(weighted_sum, activation_func):
    """Apply activation function to matrix

    :param weighted_sum: The matrix of the sum of input * weight + bias
    :type weighted_sum: numpy.ndarray
    :param activation_func: The activation function identifier
    :type activation_func: Enum.ActivationFunction
    """
    af = pick_activation(activation_func)
    if af == None:
        raise Exception("Invalid activation function")
    out = af(weighted_sum)
    #print('ACTIVATION FUNC: ', out)
    return out

def apply_loss(y, y_hat, loss_func='MSE'):
    """Apply the loss function

    :param y: the actual labels
    :type y: numpy.array
    :param y_hat: the predicted labels
    :type y_hat: numpy.array
    :param loss_func: name of the loss function, defaults to 'MSE'
    :type loss_func: str, optional
    :return: The value calculated by the loss function
    :rtype: float
    """
    loss_fn = loss_picker.get(loss_func.lower())
    return loss_fn(y, y_hat)
    
def pick_activation(activation):
    """Select activation function to use

    :param activation: The activation function identifier
    :type activation: Enum.ActivationFunction
    :return: reference to the activation function defintion
    :rtype: fn
    """
    return activation_picker.get(activation)

def enumerate_activation(activation_string):
    """Enumerate the activation function

    :param activation_string: string representing the activation function
    :type activation_string: string
    :return: The enumerated activation function
    :rtype: Enum.ActivationFunction
    """
    activation_string = activation_string.lower()
    return activation_enum.get(activation_string, ActivationFunction.NULL)


class ANN(Optimisable):
    """Fully connected Artificial Neural Network
    """
    def __init__(self):
        self.layers = []
        self.input=None
        self.y=None
        self.input_column_size=None
        self.compiled=False
        self.y_hat = None
        self.loss_fn = 'MSE'
        self.loss = None
        self.verbose_output = False


    def add(self, layer):
        """Add a new layer to the Neural Network

        :param layer: Add an instance of the layer class
        :type layer: Layer
        """
        if not inspect.isclass(Layer):
            raise Exception("Parameter must be an instance of the Layer class.")
        self.layers = self.layers + [layer]
        self.compiled = False


    def set_training_input(self, input_matrix, result_vector):
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
        self.y = result_vector    
        

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
        if self.verbose_output:
            pbar = tqdm(range(1, len(self.layers)), desc='Running model...', position=0, leave=True)
        self.layers[0].set_output(calculate_one_layer(self.input, self.layers[0]))
        # Hidden layers -> output layer
        if self.verbose_output:
            for i in pbar:
                self.layers[i].set_output(calculate_one_layer(self.layers[i-1].output, self.layers[i]))
        else:
            for i in range(1, len(self.layers)):
                self.layers[i].set_output(calculate_one_layer(self.layers[i-1].output, self.layers[i]))

        self.y_hat = np.mean(self.layers[-1].output, axis=1)#.flatten()
        self.loss = apply_loss(self.y, self.y_hat, loss_func=self.loss_fn)


    def set_loss_function(self, loss):
        """Set the loss to the specified function

        :param loss: The loss function to use
        :type loss: str
        """
        if loss_picker.has_key(loss):
            self.loss_fn = loss
        else: 
            raise ValueError('There is no loss function defined for this key')


    def vectorize(self):
        """Produce a 1D vector describing this neural network parameters

        :raises Exception: Not compiled exception
        :return: 1D vector of the neural network parameters
        :rtype: numpy.array
        """
        if not self.compiled:
            raise Exception('Compile model before producing a vector.')
        vec = []
        for layer in self.layers:
            vec.append(layer.to_vec())
        
        return np.hstack(vec)

    
    def dimension_vec(self):
        """Get a list describing the search dimensions to explore

        :return: a list of tuples each containing an upper and lower bound to search
        :rtype: list[tuple(float, float)]
        """
        dimension_vec = []
        for layer in self.layers:
            # excluding softmax which was added at a later stage, upper bound would be 6.4
            layer_vec = [[(-0.4, 5.4)]]
            if layer.use_bias:
                layer_vec.append([(-1.0, 1.0) for _ in range(layer.neurons)])
            layer_vec.append([(-1.0, 1.0) for _ in range(layer.neurons * layer.input_dimension)])
            dimension_vec += layer_vec
        return list(itertools.chain(*dimension_vec))


    def evaluate_fitness(self, vec):
        """Assess the fitness of this model given some parameters

        :param vec: An array representing all the activations, biases, and weights
        :type vec: numpy.array
        :return: a fitness score
        :rtype: float
        """
        self.decode_vec(vec)
        self.one_pass()
        return 1/ self.loss + 0.0001

    
    def decode_vec(self, vec):
        """Decode a weight vector into a neural network

        :param vec: The vector representing all the activations, biases, and weights
        :type vec: numpy.array
        """
        sub_vec = vec
        for i in range(len(self.layers)):
            input_dimension = self.layers[i].input_dimension
            columns = self.layers[i].neurons
            layer_vec_size = 1
            if self.layers[i].use_bias:
                layer_vec_size += columns
            layer_vec_size += input_dimension * columns
            self.layers[i].from_vec(sub_vec[:layer_vec_size])
            sub_vec = sub_vec[layer_vec_size:]


    def __generate_weights__(self):
        """Generates the weight matrices & bias vectors for the ANN Layers
        """

        #TODO raise exception if no input
        
        # special case for weights & bias from input layer
        self.layers[0].set_input_dimension(self.input.shape[1])
        self.layers[0].set_bias(2*np.random.rand(self.layers[0].neurons)-1)
        self.layers[0].set_weights(weight_matrix(self.layers[0].input_dimension, self.layers[0].neurons))
        
        # Construct weight matrices & bias of each layer
        for i in range(1, len(self.layers)):
            self.layers[i].set_input_dimension(self.layers[i-1].neurons)
            self.layers[i].set_bias(2*np.random.rand(self.layers[i].neurons)-1)
            self.layers[i].set_weights(weight_matrix(self.layers[i].input_dimension, self.layers[i].neurons))
        

class Layer:
    """Layer class used to add layers to the ANN
    """
    def __init__(self, neuron_count, activation="null", use_bias=True):
        """Constuctor for Layer class

        :param neurons: The number of neurons in the layer
        :type neurons: int
        :param activation: The activation funciton for the layer, defaults to "null"
        :type activation: str, optional
        :param use_bias: If you use a bias, defaults to True
        :type use_bias: bool, optional
        """
        self.neurons = neuron_count
        self.activation = enumerate_activation(activation)
        self.use_bias = use_bias

        self.input_dimension = None
        self.weights = None
        self.bias = None
        self.output = None


    def set_output(self, output):
        """Sets layer's output attributes 

        :param output: Vector of computed values from the layer
        :type output: numpy.array
        """
        self.output = output


    def set_input_dimension(self, input_dim):
        """Sets the input dimension so the layer can correctly calculate the size of the weights matrix

        :param input_dim: Number of elements from the output vector of the previous layer
        :type input_dim: int
        """
        self.input_dimension = input_dim


    def set_bias(self, bias):
        """Sets biases for the layer

        :param bias: Vector of biases for the whole layer
        :type bias: numpy.array
        """
        self.bias = bias
    

    def set_weights(self, weights):
        """Sets weights for the layer

        :param weights: Matrix of weights for the layer
        :type weights: numpy.array
        """
        self.weights = weights
    

    def to_vec(self):
        """Convert the layer into a vector

        :return: layer vector
        :rtype: numpy.array
        """
        if self.use_bias:
            t = np.append(int(self.activation), self.bias.flatten())
            return np.append(t, self.weights.flatten())
        else:
            return np.append(int(self.activation), self.weights.flatten())


    def from_vec(self, vec):
        """Convert a vector into a layer

        :param vec: a vector describing the layers activation function, bias, and weights
        :type vec: numpy.array
        """
        self.activation = ActivationFunction(round(vec[0]))
        weights_position = 1
        if self.use_bias:
            self.bias = vec[1:self.neurons+1]
            weights_position += len(self.bias)
        self.weights = vec[weights_position:].reshape(self.input_dimension, self.neurons)
