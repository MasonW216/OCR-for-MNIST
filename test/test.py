import numpy as np
import pandas as pd

data = pd.read_csv("/Users/masonwang/Documents/Code/MNIST OCR ML/OCR-for-MNIST/data/mnist_test.csv")
data = np.array(data)
m, n = data.shape
test_data = data[0:1000].T
test_y = test_data[0]
test_x = test_data[1:n]

training_data = data[1000:m].T
training_y = training_data[0]
training_x = training_data[1:n]

print(training_x)
print(m,n)
print("\n")
print(training_x.shape)
