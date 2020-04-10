import numpy as np


def perceptron_train(X, d):
    w0 = 1.0
    w = np.ones(len(X[0]), float)

    ALPHA = 0.9
    MAX_EPOCHS = 20

    for epoch in range(MAX_EPOCHS):
        any_misclassified = False
        for i in range(len(X)):
            # print("i =", i)                                   # tmp
            s = np.dot(X[i], w) + w0
            z = 1.0 if s > 0 else -1.0
            # print(format(s, '8.3f'), format(z, '4.1f'))       # tmp
            if z  !=  d[i]:
                w0 = w0 + ALPHA * d[i]
                w = w + ALPHA * X[i] * d[i]
                #print(w0, w)                                   # tmp
                any_misclassified = True
        if not any_misclassified:
            print("Learned in", epoch, "epochs.")               # tmp
            break
    else:
        print("Stopped at", MAX_EPOCHS, "epochs.")
    # print("w:\n", w)
    # print("w0: ", w0)
    return w0, w 


def perceptron_use(w0, w, X):
    z = np.zeros(len(X), float)
    for i in range(len(X)):
        s = np.dot(X[i], w) + w0
        z[i] = 1.0 if s > 0 else -1.0
    return(z)


if __name__ == "__main__":
    X = np.array([
        [-2,  4],
        [ 4,  1],
        [ 1,  6],
        [ 2,  4],
        [ 6,  2],])
    d = np.array([-1, -1, 1, 1, 1])

    w0, w = perceptron_train(X, d)
    print(w0, w)
    z = perceptron_use(w0, w, X)
    print(z - d)