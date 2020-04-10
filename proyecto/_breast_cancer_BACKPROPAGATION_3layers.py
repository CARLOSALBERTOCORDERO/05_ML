# ann_1_.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class ANN:
    def _sigmoid(self, x):
       return 1.0 / (1.0 + np.e ** (-x))

    def _sig(self, x):
       return np.sign(x)

    def _step(self, x):
        return np.heaviside(x, 0.5)

    def __init__(self, input_n, hidden_n, hidden1_n, output_n, ALPHA=0.3):
        # Number of nodes per layer
        self.I = input_n
        self.J = hidden_n
        self.K = hidden1_n;
        self.L = output_n
        # Learning rate
        self.lr = ALPHA

        # Weight matrices
        self.wij = np.random.rand(self.J, self.I)
        self.wjk = np.random.rand(self.K, self.J)
        self.wkl = np.random.rand(self.L, self.K)

        # activation functions
        self.activation_h = self._sigmoid
        self.activation_h1 = self._sigmoid
        self.activation_o = self._step
        
    def train(self, input_vector, target_vector):
        # convert received lists into 2d arrays
        inputs = np.array(input_vector, ndmin=2).T
        # print("X:\n", inputs)
        targets = np.array(target_vector, ndmin=2).T
        # print("T:\n", targets)
        
        # calculate internal states and activation of hidden neurons
        hidden_states = np.dot(self.wij, inputs)
        # print("HS\n", hidden_states)
        hidden_activations = self.activation_h(hidden_states)
        # print("Y\n", hidden_activation)
        print(hidden_activations)

        
        # calculate internal states and activation of output neurons
        hidden1_states = np.dot(self.wjk, hidden_activations)
        # calculate the signals emerging from final output layer
        hidden1_activations = self.activation_h1(hidden1_states)

        # calculate internal states and activation of output neurons
        output_states = np.dot(self.wkl, hidden1_activations)
        # calculate the signals emerging from final output layer
        output_activations = self.activation_o(output_states)
        
        # Output layer errors
        output_errors = output_activations - targets
        #print("OE:", output_errors)

        # Hidden1 layer errors
        hidden1_errors = np.dot(self.wkl.T, output_errors)

        # Hidden layer errors
        hidden_errors = np.dot(self.wjk.T, hidden1_errors)

        # Update weights in the output neurons
        self.wkl -= self.lr * np.dot((output_errors *
                output_activations * (1.0 - output_activations)),
                np.transpose(hidden1_activations))

        # Update weights in the hidden1 neurons
        self.wjk -= self.lr * np.dot((hidden1_errors *
                hidden1_activations * (1.0 - hidden1_activations)),
                np.transpose(hidden_activations))

        # Update weights in the hidden neurons
        self.wij -= self.lr * np.dot((hidden_errors *
                hidden_activations * (1.0 - hidden_activations)), 
                np.transpose(inputs))


    # get output from the traiend network
    def get_output(self, input_vector):
        # print("Use")
        # convert inputs list to 2d array
        inputs = np.array(input_vector, ndmin=2).T
        # print("IN:\n", inputs)
        # calculate signals into hidden layer
        hidden_state = np.dot(self.wij, inputs)
        # print("HS:\n", hidden_state)
        # calculate the signals emerging from hidden layer
        hidden_activation = self.activation_h(hidden_state)
        # print("HY:\n", hidden_activation)
        # calculate signals into final output layer
        hidden1_state = np.dot(self.wjk, hidden_activation)
        # print("HS:\n", hidden_state)
        # calculate the signals emerging from hidden layer
        hidden1_activation = self.activation_h(hidden1_state)
        # print("HY:\n", hidden_activation)
        # calculate signals into final output layer
        output_states = np.dot(self.wkl, hidden1_activation)
        # print("OS\n", output_states)
        # calculate the signals emerging from final output layer
        output_activations = self.activation_o(output_states)
        # print("OZ:\n", outputs)
        return output_activations 

def main(verbose=True, filename="breast-cancer.csv", cycles=2500000):
    np.random.seed(13)
    breastCancerData = pd.read_csv(filename).dropna()
    breastCancerX = pd.get_dummies(breastCancerData.drop('CellClass_2_benign4_malignant', axis=1))
    breastCancerX['bias'] = 1
    breastCancerX = breastCancerX.values
    ##breastCancerY = (np.atleast_2d(breastCancerData['CellClass_2_benign4_malignant']).T == '4').astype(int)
    breastCancerY = np.atleast_2d(breastCancerData['CellClass_2_benign4_malignant']).T
    for index in range(len(breastCancerY)):
        if(4 == breastCancerY[index][0]):
            breastCancerY[index][0] = 1;
        else:
            breastCancerY[index][0] = 0;


    x_train, x_test, y_train, y_test = train_test_split(breastCancerX, breastCancerY, train_size=0.85, test_size=0.15)
    # number of input, hidden and output nodes
    inp_nodes = len(breastCancerX[0])
    hid_nodes = 32
    hid1_nodes = 16
    out_nodes = 1

    # learning rate
    alpha = 0.30

    # create instance of neural network
    ann = ANN( inp_nodes, hid_nodes,hid1_nodes, out_nodes, alpha)

    inputs = np.array(x_train, float)
    targets = np.array(y_train, float)
    EPOCHS = cycles
    for E in range(EPOCHS+1):
        ann.train(inputs, targets)
        if (True == verbose):
            print(format(E / EPOCHS, '8.4%'))

    # test get_output_for (doesn't mean anything useful yet)
    if (True == verbose):
        for index in range(len(x_test)):
            print("OUT:\n", ann.get_output(x_test[index]))
            print("Expected:\n", y_test[index][0])


if __name__ == "__main__":
    main()