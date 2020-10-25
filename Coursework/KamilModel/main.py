from KamilModel import *
from data import Data
import numpy
import pandas as pd

#get data
data = Data("2in_xor.txt")

#get matrix of input
data_x = data.get_rows()

#create neural network
ann = ANN()

# create layers
ann.set_input(data_x, use_bias=True)
ann.add(Layer(3, "sigmoid", use_bias=True))
ann.add(Layer(2, "sigmoid", use_bias=True))
#last layer cretaed is the output layer
ann.add(Layer(1, "sigmoid"))

ann.compile()
output = ann.epoch()

print(output.neurons_val)

