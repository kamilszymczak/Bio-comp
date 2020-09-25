from enum import Enum
import numpy as np

# Enumerated activation functions
ActivationFunction = Enum('Activation_function', 'NULL SIGMOID HYPERBOLIC_TANGENT COSINE GAUSSIAN')

class ANN:
    """Artificial Neural Network(ANN) class.
    Use to build and run your ANN."""
    
    def __init__(self) -> None:
        """Initialise unconfigured ANN."""
        # TODO: determine structure of layers (e.g. type like: List[Layer])
        self.layers = []
        # TODO: Determine stucture of input (e.g. type like: np.ndarray[float64])
        self.inputs = []
        self.activation_function = ActivationFunction.NULL   

    def set_activation_func(self, af: ActivationFunction) -> None:
        """Define the activation function the neurons in this network will use, defaults to NULL."""
        self.activation_function = af

    def set_input(self, input) -> None:
        """Set the input matrix for the neural network"""
        self.inputs = input

    # TODO: implement function
    def append_layer(self, layer) -> None: 
        """Add a new layer to the neural network."""
        # TODO: Implement structure of layers. With append functionality
        self.layers = layer

    # TODO: implement function
    def count_layers(self) -> int:
        """Return the number of layers"""
        return 0

    # TODO: implement function
    def count_neurons(self) -> int:
        """Return the number of neurons"""
        return 0

    # TODO: implement train
    def train(self):
        """Train the ANN using the config settings stored"""
        return None

    # TODO: implement infer
    def infer(self):
        """Use the ANN to make predictions based on the current weights"""
        return None

    # TODO: figure out more functionality for ANN

class Layer:
    """Layer class for use in ANN"""
    def __init__(self) -> None:
        self.neurons = []

    """Adds a `neuron` `num_add` times to the layer"""
    def add_neuron(self, num_add, neruon) -> None:
        return None

    # TODO: figure out more functionality for Layer

class Neuron:
    """Neuron class, Layer objects use Neurons"""
    def __init__(self) -> None:
        self.weights = np.zeros(0)

    # TODO: figure out more functionality for Neuron

    




