import numpy as np
import pandas as pd

data = pd.read_csv("/Users/masonwang/Documents/Code/MNIST OCR ML/OCR-for-MNIST/data/mnist_test.csv")
data = np.array(data)
print(data,"\n")
m, n = data.shape
test_data = data[0:1000].T
y = test_data[0]
x = test_data[1:n]
print(m,n)
print(y,"\n")
print(x)