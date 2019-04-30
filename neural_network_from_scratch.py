import numpy as np
from matplotlib import pyplot as plt

#TODO: Get more data
data = [['000000', 0], ['FFFFFF', 1], ['FFFAFA', 1], ['F0FFF0', 1],
        ['F5FFFA', 1], ['F0FFFF', 1], ['F0F8FF', 1], ['F8F8FF', 1],
        ['F5F5F5', 1], ['FFF5EE', 1], ['F5F5DC', 1], ['FDF5E6', 1],
        ['FFFAF0', 1], ['FFFFF0', 1], ['FAEBD7', 1], ['FAF0E6', 1]]

class NN:
    """Neural Network with one hidden layer"""
    
    def __init__(self):
        self.w1 = np.random.randn(3, 5)  #first weight matrix
        self.b1 = np.random.randn(1, 5)  #first set of bias
        self.w2 = np.random.randn(5, 2)  #second weight matrix
        self.b2 = np.random.randn(1, 2)  #second set of bias
        self.learning_rate = 0.3
        
    @staticmethod 
    def sigmoid(x):
        """Compute the sigmoid of vector x."""
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def softmax(x):
        """Compute the softmax of vector x."""
        return np.exp(x)/np.sum(np.exp(x))

    def sigmoid_prime(self, x):
        """Compute the derivative of sigmoid for vector x."""
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    @staticmethod
    def colorConvert(colorCode): 
        """Converts a hex color code to its red, blue and green values."""
        r = int(colorCode[:2], 16)
        g = int(colorCode[2:4], 16)
        b = int(colorCode[4:], 16)
        return r, g, b
    
    def forward(self, color):
        """Gets the prediction of a single data input. Intended for model"""
        r, g, b = self.colorConvert(color)
        self.x = np.array([[r, g, b]])
        self.z1 = np.add(np.matmul(self.x, self.w1), self.b1)   #hidden layer nodes before sigmoid
        self.a1 = self.sigmoid(self.z1)                    #hidden layer nodes final value
        self.z2 = np.add(np.matmul(self.a1, self.w2), self.b2)  #output before sigmoid
        a2 = self.sigmoid(self.z2)
        return(a2)
        
    def predict(self, color):
        """Gets the prediction of a single data input. Intended for user"""
        r, g, b = self.colorConvert(color)
        self.x = np.array([[r, g, b]])
        self.z1 = np.add(np.matmul(self.x, self.w1), self.b1)   #hidden layer nodes before sigmoid
        self.a1 = self.sigmoid(self.z1)                    #hidden layer nodes final value
        self.z2 = np.add(np.matmul(self.a1, self.w2), self.b2)  #output before sigmoid
        a2 = self.sigmoid(self.z2)
        return(np.argmax(a2))
        
    def fit(self, data):
        """ """
        costs = []
        
        for i in range(5000):
            random_data_point = data[np.random.randint(len(data))]
            y = random_data_point[1]
            pred = self.forward(random_data_point[0])
            cost = np.sum(np.square(y - pred))
            costs.append(cost)
            
            #All partical derivatives
            dc_da2 = 2 * (pred - y)
            da2_dz2 = self.sigmoid_prime(self.z2)
            dz2_dw2 = self.a1
            dz2_db2 = 1
            dz2_da1 = self.w2
            da1_dz1 = self.sigmoid_prime(self.z1)
            dz1_dw1 = self.x
            dz1_db1 = 1
            
            #Gradients
            dc_dw2 = dz2_dw2.T * (dc_da2 * da2_dz2)
            dc_db2 = dc_da2 * da2_dz2 * dz2_db2
            dc_dw1 = np.matmul(dz1_dw1.T * (dc_da2 * da2_dz2), (dz2_da1.T * da1_dz1))
            dc_db1 = np.matmul(dz1_db1 * (dc_da2 * da2_dz2), (dz2_da1.T * da1_dz1))
            
            #Adjust weights and bias
            self.w1 = self.w1 - self.learning_rate * dc_dw1
            self.b1 = self.b1 - self.learning_rate * dc_db1
            self.w2 = self.w2 - self.learning_rate * dc_dw2
            self.b2 = self.b2 - self.learning_rate * dc_db2
            
        plt.clf()
        plt.plot(costs)
        
        
    
