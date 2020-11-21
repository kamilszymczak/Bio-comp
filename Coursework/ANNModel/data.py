import pandas as pd
import numpy as np

class Data:
    """Data class to handle normalization and structure of input data
    """

    # TODO: specify number of features or detect automatically? 
    def __init__(self, infile, normalize=False, delim="\t") -> None:
        """Constructor for Data class

        :param infile: String of filename
        :type infile: String
        :param normalize: To normalize this data, defaults to False
        :type normalize: bool, optional
        :param delim: Delimiter for reading data
        :type delim: String
        """
        self.df = pd.DataFrame(np.loadtxt(infile, dtype=float))

        if normalize:
            self.normalize()


    def normalize(self) -> None:
        self.df = (self.df-self.df.min()) / (self.df.max()-self.df.min())

    # returns all features except the outcome as a numpy matrix
    def get_rows(self):
        """Get all the rows of the datafile, excluding the outcome column

        :return: returns numpy ndarray of input data
        :rtype: numpy.array
        """
        return self.df.drop(self.df.columns[-1],axis=1).to_numpy()

    # returns outcome column as a numpy matrix
    def get_output(self):
        """Desired result vector

        :return: numpy ndarray with the actual results
        :rtype: numpy.array
        """
        return self.df.iloc[:,-1].to_numpy()

    def score(self, model, test_labels, data_name):
        """Accuracy Scores of model

        :param model: ANN model 
        :type model: ANNModel.Model.ANN
        :param test_labels: actual test data outcomes
        :type test_labels: numpy.array
        :param data_name: Name of dataset
        :type data_name: string
        :return: tuple of scores 
        :rtype: tuple(float)
        """
        correctly_classified_A, correctly_classified_B = 0, 0

        decimal_palces = {
            'cubic': 4,
            'linear': 2,
            'sine': 4,
            'tanh': 4,
            'complex': 4,
            'xor': 0
        }

        #get predictions
        predictions = np.round(model.y_hat, decimal_palces.get(data_name))
        # print(model.y)
        # print(predictions)

        #loop through predictions
        for index, pred in enumerate(predictions):

            if np.isclose(test_labels[index], pred):
                correctly_classified_A += 1

            if np.isclose(pred, test_labels[index]):
                correctly_classified_B += 1

        return correctly_classified_A/len(test_labels), correctly_classified_B/len(test_labels)     