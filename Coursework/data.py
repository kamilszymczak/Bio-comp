import pandas as pd

class Data:

    # TODO: specify number of features or detect automatically? 
    def __init__(self, infile, normalize=False) -> None:
        self.df = pd.read_csv(infile, header=None, delimiter="\t", na_values='.') #read the data

        if normalize:
            self.normalize()


    def normalize(self) -> None:
        self.df = (self.df-self.df.min()) / (self.df.max()-self.df.min())

    # returns all features except the outcome as a numpy matrix
    def getX(self):
        return self.df.iloc[:,:-1].to_numpy()

    # returns outcome column as a numpy matrix
    def getY(self):
        return self.df.iloc[:,-1].to_numpy()



    