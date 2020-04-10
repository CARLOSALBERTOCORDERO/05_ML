# Francico Javier Delgadillo Casas
# MDE705055
# Final Project Logistics Regression with Numpy

import numpy as np
from sklearn.metrics import f1_score

def sigmoid(z):
    return 1 / (1 + np.e ** (-z))


def calc_probability(b, W, el):
    return np.e ** (b + np.dot(el, W)) / (1 + np.e ** (b + np.dot(el, W)))


def logistic_loss(y, y_hat):
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def main(verbose=True, filename="breast-cancer.csv", cycles=100000):
    data_file = open(filename, 'r')
    data_list = data_file.readlines()
    data_file.close()
    data = []
    for i in range(len(data_list)):
        if i == 0:
            headers = data_list[i].split(',')
        else:
            line = data_list[i].split(',')
            line = [int(el.rstrip('\n')) for el in line]
            data.append(line)
    data_len = len(data[0]) - 1
    data_np = np.array([np.array(xi) for xi in data])
    X = data_np[:, :data_len]  # Number of independant variables to take in account
    y = data_np[:, data_len]

    for el in range(0, len(y)):
        if y[el] == 4:
            y[el] = 1
        else:
            y[el] = 0

    perm = np.random.permutation(len(data))
    x_train, x_test = X[perm][150:], X[perm][:150]
    y_train, y_test = y[perm][150:], y[perm][:150]

    if verbose:
        print(x_train, x_test)
        print(len(x_train), len(x_test))
        print(y_train, y_test)
        print(len(y_train), len(y_test))

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    W = np.zeros((data_len, 1))
    b = np.zeros((1, 1))
    learning_rate = 0.03

    m = len(y_train)
    print(m, x_train)

    for epoch in range(cycles):
        Z = np.matmul(x_train, W) + b
        A = sigmoid(Z)
        loss = logistic_loss(y_train, A)
        dz = A - y_train
        dw = 1 / m * np.matmul(x_train.T, dz)
        db = np.sum(dz)
        W = W - learning_rate * dw  # B1 and B2 from the logistic reggresion summary
        b = b - learning_rate * db  # B0 from the logistic regression summary
        if (epoch % 1000 == 0) and verbose:
            print("Error= " + str(loss))

    if verbose:
        print("Weights and bias for formula Y = b + W0 * e0 + W1 * e1 + ..... + Wn * en")
        print("Written in previous formula would result on an estimated Y")
        print("B Value = ")
        print(b)
        print("W value = ")
        print(W)

    ## Testing results
    pred = []
    pred_prob = []
    for el in x_test:
        temp = calc_probability(b, W, el)
        pred_prob.append(float(temp) * 100)
        if temp > 0.5:
            pred.append(1)
        else:
            pred.append(0)
    print("Predicted Probabilities Values: ")
    print(pred_prob)
    print("Predicted Round Values: ")
    print(pred)
    print("Actual Values Values: ")
    print(*y_test)
    print("Percentage of correct computations: {0:.0%}".format(f1_score(pred, y_test)))

if __name__ == "__main__":
    main()