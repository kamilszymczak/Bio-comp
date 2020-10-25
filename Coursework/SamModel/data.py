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
        :rtype: ndarray
        """
        return self.df.drop(self.df.columns[-1],axis=1).to_numpy()

    # returns outcome column as a numpy matrix
    def get_output(self):
        """Desired result vector

        :return: numpy ndarray with the actual results
        :rtype: ndarray
        """
        return self.df.iloc[:,-1].to_numpy()



    