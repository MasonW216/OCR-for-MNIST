import sys
sys.path.append("/Users/masonwang/Documents/Code/MNIST OCR ML/OCR-for-MNIST/src")
import numpy as np
import pandas as pd
from data_loader import load_mnist, normalise
from neural_network import neural_network
test_y, test_x, training_y, training_x, m, n = load_mnist("/Users/masonwang/Documents/Code/MNIST OCR ML/OCR-for-MNIST/data/mnist_test.csv")
test_x, training_x = normalise(test_x), normalise(training_x)

mnist_ocr = neural_network(784, 128, 10, 10, 0.1)
mnist_ocr.train(training_x,training_y,100)
predictions, accuracy = mnist_ocr.predict(test_x,test_y)
print(test_y)
print(predictions)
print(accuracy)