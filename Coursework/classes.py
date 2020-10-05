from enum import Enum
import numpy as np
from data import Data

# Enumerated activation functions
ActivationFunction = Enum('ActivationFunction', 'NULL SIGMOID HYPERBOLIC_TANGENT COSINE GAUSSIAN')

LayerConnection = Enum('LayerConnection', 'FULL SPARSE')

LearningStrategy = Enum('LearningStrategy', 'BATCH SEMI-BATCH LIVE')

class ANN:
    """Artificial Neural Network(ANN) class.
    Use to build and run your ANN."""

    # Fields to think about:
    # Learning rate
    # Loss convergence
    # structure of Input vector
    # static neuron config
    # batch / live / semi-batch learning
    
    def __init__(self) -> None:
        """Initialise unconfigured ANN."""
        # TODO: determine structure of layers (e.g. type like: List[Layer])
        self.layers = []
        # TODO: Determine stucture of input (e.g. type like: np.ndarray[float64])
        self.inputs = []   
        # TODO: Create enum of learning strategy
        self.learning_strategy = LearningStrategy.BATCH
        # Iterations? loss convergence? maybe ANN should be abstract, and different strategies could implement ANN

    #! -------------------- BUILDER FUNCTIONS --------------------

    # TODO: implement function
    def append_layer(self, layer) -> None:
        """Add a new layer to the neural network."""
        # TODO: Implement structure of layers. With append functionality
        self.layers = layer

    # TODO: implement function
    def connect_layers(self) -> None:
        """Connect all the layers inside the ANN"""
        return 0


    #! -----------------------------------------------------------



    def set_input(self, input) -> None:
        """Set the input matrix for the neural network"""
        self.inputs = input

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

    # helper function for train()
    def update_weights(self):
        """Perform back propagation to update weights"""
        return None

    # TODO: implement infer
    def infer(self):
        """Use the ANN to make predictions based on the current weights"""
        return None

    # TODO: figure out more functionality for ANN
    # Nice to haves:
    # Might be cool to implement some form of pruning, (remove weights close to 0)
    # Or quantization (convert tree from float64 to something like float 8 or smaller)

class Layer:
    """Layer class for use in ANN, instantiate and empty Layer"""
    def __init__(self) -> None:
        self.neurons = []
        self.activation_function = ActivationFunction.NULL

    """Adds a `neuron` `layer_size` times to the layer"""
    def add_neuron(self, layer_size, neruon) -> None:
        return None

    def set_activation_func(self, af: ActivationFunction) -> None:
        """Define the activation function the neurons in this network will use, defaults to NULL."""
        self.activation_function = af

    # TODO: figure out more functionality for Layer

class Neuron:
    """Neuron class, Layer objects use Neurons"""
    def __init__(self) -> None:
        # Weights will be a vector of float64 values, length of this vector is equal to the number of neurons in the previous layer 
        self.weights = np.zeros(0)

    # TODO: figure out more functionality for Neuron

    




