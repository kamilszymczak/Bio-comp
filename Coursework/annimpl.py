import numpy as np

    """Artificial Neural Network Implementation
    """

def __weight_matrix__(x, y):
        """Return a numpy array of random weights with the specified size

        :param x: number of rows in the desired matrix
        :type x: int
        :param y: number of columns in the desired matrix
        :type y: int
        """
        return 2*np.random.rand(x, y)-1

class ANN:
    def __init__(self, input_matrix, result_vector, layer_def=[3, 2]):
        """Construct a new Fully Connected Artificial Neural Network

        :param layer_def: Define the number of neurons in each layer (not including the input layer), defaults to [3, 2]
        :type layer_def: list[int], optional
        :param input_matrix: The input matrix used to train the ANN
        :type input_matrix: numpy.ndarray
        :param result_vector: The Result vector used to determine the error of the network output
        :type result_vector: numpyndarray
        """
        self.input_matrix = input_matrix
        self.result_vector = result_vector
        self.layer_def = layer_def
        

    def generate_weights(self):
        """Generates the weight matrices for the ANN

        :param layer_def: The number of neurons in each layer (not including the input layer), defaults to self.layer_def
        :type layer_def: list[int], optional
        """
        num_rows = self.input_matrix.shape[0]
        neuron_definition = [num_rows] + self.layer_def
        
        temp_weights = []
        for i  in range(1, len(neuron_definition)):
            temp_weights = temp_weights + [__weight_matrix__(neuron_definition[i-1], neuron_definition[i])]
        self.weights = temp_weights

    def layer_output(self, neurons_matrix, weights_matrix):
        """Return a numpy array of layer output, input for next layer's neurons

        :param neurons_matrix: matrix of all neurons from previous layer
        :type neurons_matrix: numpy.ndarray
        :param weights_matrix: matrix of weights
        :type weights_matrix: numpy.ndarray
        """
        return np.dot(neurons_matrix, weights_matrix)

    
        

