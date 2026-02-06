import numpy as np
import pandas as pd

#reading the mnist dataset .csv file into a numpy array
#separates test data from training data
def load_mnist(filepath):
    data = pd.read_csv(filepath)
    data = np.array(data)
    m, n = data.shape
    test_data = data[0:1000].T
    test_y = test_data[0]
    test_x = test_data[1:n]

    training_data = data[1000:m].T
    training_y = training_data[0]
    training_x = training_data[1:n]

    return test_y, test_x, training_y, training_x, m, n

def normalise(data_x):
    return data_x/255.0

test_y, test_x, training_y, training_x = load_mnist("/Users/masonwang/Documents/Code/MNIST OCR ML/OCR-for-MNIST/data/mnist_test.csv")
ntraining_x = normalise(training_x)