import numpy as np

class neural_network:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.learning_rate = learning_rate

    def generate_weights(self):
        self.w1 = np.random.randn(self.hidden1_size,self.input_size) * np.sqrt(2/self.input_size)
        self.w2 = np.random.randn(self.hidden2_size,self.hidden1_size) * np.sqrt(2/self.hidden1_size)
        self.w3 = np.random.randn(self.output_size,self.hidden2_size) * np.sqrt(2/self.hidden2_size)

    def generate_biases(self):
        self.b1 = np.zeros((self.hidden1_size,1))
        self.b2 = np.zeros((self.hidden2_size,1))
        self.b3 = np.zeros((self.output_size,1))
    
    def forward_prop(self,data):
        self.a_layer0 = data
        self.u_hidden1 = self.w1 @ data + self.b1
        self.a_hidden1 = ReLU(self.u_hidden1)
        self.u_hidden2 = self.w2 @ self.a_hidden1 + self.b2
        self.a_hidden2 = ReLU(self.u_hidden2)
        self.u_layer3 = self.w3 @ self.a_hidden2 + self.b3
        self.output = softmax(self.u_layer3)
        return self.output
    
    def backward_prop(self,output,labels):
        m,n = output.shape
        ohe_labels = np.zeros((self.output_size,n))
        ohe_labels[labels, np.arange(n)] = 1
        self.dz3 = output - ohe_labels
        self.dw3 = 1/n * self.dz3 @ self.a_hidden2.T
        self.db3 = 1/n * np.sum(self.dz3, axis=1, keepdims=True)
        self.dz2 = self.w3.T @ self.dz3 * ReLUderimask(self.u_hidden2)
        self.dw2 = 1/n * self.dz2 @ self.a_hidden1.T
        self.db2 = 1/n * np.sum(self.dz2, axis=1, keepdims=True)
        self.dz1 = self.w2.T @ self.dz2 * ReLUderimask(self.u_hidden1)
        self.dw1 = 1/n * self.dz1 @ self.a_layer0.T
        self.db1 = 1/n * np.sum(self.dz1, axis=1, keepdims=True)

    def update_params(self):
        self.w1 -= self.learning_rate * self.dw1
        self.w2 -= self.learning_rate * self.dw2
        self.w3 -= self.learning_rate * self.dw3
        self.b1 -= self.learning_rate * self.db1
        self.b2 -= self.learning_rate * self.db2
        self.b3 -= self.learning_rate * self.db3
    
    def predict(self,test_data,test_labels):
        output = self.forward_prop(test_data)
        predictions = np.argmax(output, axis=0)
        accuracy = np.mean(predictions == test_labels)
        return predictions, accuracy
    
    def train(self,data,labels,iterations):
        self.generate_weights()
        self.generate_biases()
        for i in range(iterations):
            self.forward_prop(data)
            self.backward_prop(self.output,labels)
            self.update_params()
#creating activation function
def ReLU(nn_layer):
    nn_layer = np.maximum(0,nn_layer)
#more rudimentary version
#   m, n = nn_layer.shape
#   for i in range(0,m):
#       for j in range(0,n):
#           if nn_layer[i,j] < 0:
#               nn_layer[i,j] = 0
    return nn_layer

def ReLUderimask(nn_layer):
    m, n = nn_layer.shape
    mask = np.zeros((m,n))
    for i in range(0,m):
        for j in range(0,n):
            if nn_layer[i,j] <= 0:
                mask[i,j] = 0
            else:
                mask[i,j] = 1
    return mask

def softmax(nn_layer):
    probs = np.exp(nn_layer) / np.sum(np.exp(nn_layer), axis=0, keepdims=True)
    return probs

mnist_ocr = neural_network(784,128,10,10,0.1)
mnist_ocr.generate_weights()
mnist_ocr.generate_biases()