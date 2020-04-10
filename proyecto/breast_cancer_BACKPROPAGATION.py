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
        threshold = 0.5
        return np.heaviside(x - threshold, 1.0)

    def __init__(self, input_n, hidden_n, output_n, ALPHA=0.3):
        # Number of nodes per layer
        self.I = input_n
        self.J = hidden_n
        self.K = output_n
        # Learning rate
        self.lr = ALPHA

        # Weight matrices
        self.wij = np.random.rand(self.J, self.I)
        self.wjk = np.random.rand(self.K, self.J+1)
        ##self.wjk = np.random.rand(self.K, self.J)

        # activation functions
        self.activation_h = self._sigmoid
        self.activation_o = self._sigmoid
        
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
        #print(hidden_activations)
        hidden_activations = np.insert(hidden_activations, 0,1, axis=0)
        #print(hidden_activations)
        #
        
        # calculate internal states and activation of output neurons
        final_states = np.dot(self.wjk, hidden_activations)
        # calculate the signals emerging from final output layer
        output_activations = self.activation_o(final_states)
        
        # Output layer errors
        output_errors = output_activations - targets
        #print("OE:", output_errors)

        # Hidden layer errors
        hidden_errors = np.dot(self.wjk.T, output_errors) 
        
        # Update weights in the output neurons
        self.wjk -= self.lr * np.dot((output_errors * 
                output_activations * (1.0 - output_activations)),
                np.transpose(hidden_activations))
        
        derivation = hidden_activations * (1.0 - hidden_activations)
        derivation = derivation * derivation
        derivation = np.delete(derivation,0,0)
        
        # Update weights in the hidden neurons
        hidden_errors_in = np.delete(hidden_errors,0,0)
        self.wij -= self.lr * np.dot((hidden_errors_in *
                derivation), 
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
        hidden_activation = np.insert(hidden_activation, 0,1, axis=0)
        output_states = np.dot(self.wjk, hidden_activation)
        # print("OS\n", output_states)
        # calculate the signals emerging from final output layer
        output_activations = self.activation_o(output_states)
        # print("OZ:\n", outputs)
        return output_activations 

def main(verbose=True, filename="_breast-cancer.csv", cycles=1000):
    np.random.seed(13)
    breastCancerData = pd.read_csv(filename).dropna()
    breastCancerX = pd.get_dummies(breastCancerData.drop('CellClass_2_benign4_malignant', axis=1))
    breastCancerX['bias'] = 1
    breastCancerX = breastCancerX.values
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
    out_nodes = 1

    # learning rate
    alpha = 0.3

    # create instance of neural network
    ann = ANN( inp_nodes, hid_nodes, out_nodes, alpha)

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
            print("OUT: ", ann.get_output(x_test[index]))
            print("Expected: ", y_test[index][0])


if __name__ == "__main__":
    main()