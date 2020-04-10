# Cordero Robles, Carlos Alberto
# https://blog.goodaudience.com/logistic-regression-from-scratch-in-numpy-5841c09e425f
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt



def main():
    # Take database
    iris = load_iris()
    X = iris.data
    Y = iris.target
    non_versicolor_labels = [0 for i in range(100)]
    versicolor_labels = [1 for i in range(50)]
    y = np.concatenate([non_versicolor_labels, versicolor_labels])
    perm = np.random.permutation(150)
    # Separate database in test and train
    x_train, x_test = X[perm][20:], X[perm][:20]
    y_train, y_test = Y[perm][20:], Y[perm][:20]
    # Reshape info to make it easy to manipulate
    # Y transformed to a column and X just take 2 values
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    x_train = x_train[:, 2:]
    x_test = x_test[:, 2:]
    # Plot info
    #plt.plot(x_train[versicolor][:,0], x_train[versicolor][:,1], 'b.')
    #plt.plot(x_train[not_versicolor][:,0], x_train[not_versicolor][:,1], 'r.')
    #plt.xlabel('Petal Length (cm)')
    #plt.ylabel('Petal With (cm)')
    #plt.legend(['Versicolor', 'Not Versicolor'])
    #plt.show()
    
    #sinoid plot
    
    W = np.zeros((2,1))
    b = np.zeros((1,1))
    learning_rate = 0.01
    m = len(y_train)
    
    for epoch in range(5000):
        Z = np.matmul(x_train,W) + b
        A = sigmoid(Z)
        loss = logistic_loss(y_train, A)
        dz = A - y_train
        dw = 1/m * np.matmul(x_train.T,dz)
        db = np.sum(dz)
        
        W = W - learning_rate * dw
        b = b - learning_rate * db
        
        if epoch % 100 == 0:
            print(loss)
    
    
def sigmoid(z):
    return 1 / (1 + np.e**(-z))
    
def logistic_loss(y, y_hat):
    return -np.mean(y * np.log(y_hat) + (1-y) * np.log(1 - y_hat))
    

    
if __name__ == "__main__":
    main()