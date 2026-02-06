import numpy as np
import pandas as pd

def load_mnist(filepath):
    data = pd.read_csv(filepath)
    data = np.array(data)
    m, n = data.shape
    test_data = data[0:1000].T
    y = test_data[0]
    x = test_data[1:n]
    return x, y

print(load_mnist("/Users/masonwang/Documents/Code/MNIST OCR ML/OCR-for-MNIST/data/mnist_test.csv"))
