# ann_1_.py
import numpy as np


class ANN:
    def _sigmoid(self, x):
       return 1.0 / (1.0 + np.e ** (-x))
    
    def __init__(self, input_n, hidden_n, output_n, ALPHA=0.3):
        # Number of nodes per layer
        self.I = input_n
        self.J = hidden_n
        self.K = output_n
        # Learning rate
        self.lr = ALPHA

        # Weight matrices
        self.wij = np.random.rand(self.J, self.I)
        self.wjk = np.random.rand(self.K, self.J)

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
        # print("Y\n", hidden_activation)
        
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
        output_states = np.dot(self.wjk, hidden_activation)
        # print("OS\n", output_states)
        # calculate the signals emerging from final output layer
        output_activations = self.activation_o(output_states)
        # print("OZ:\n", outputs)
        return output_activations 

      
if __name__ == "__main__":
    np.random.seed(13)

    # number of input, hidden and output nodes
    inp_nodes = 3
    hid_nodes = 5
    out_nodes = 2

    # learning rate
    alpha = 0.25

    # create instance of neural network
    ann = ANN(inp_nodes, hid_nodes, out_nodes, alpha)

    inputs = np.array([[1, 2, 7],
                       [1, 3, 7], 
                       [1, 4, 7], 
                       [1, 5, 7], 
                       [1, 6, 7]], float)
    targets = np.array([[1/6, 5/6], 
                        [2/6, 4/6], 
                        [3/6, 3/6], 
                        [4/6, 2/6], 
                        [5/6, 1/6]], float)
    EPOCHS = 10_000
    for E in range(EPOCHS+1):
        ann.train(inputs, targets)
        print(format(E / EPOCHS, '8.4%'))

    # test get_output_for (doesn't mean anything useful yet)
    print("OUT:\n", ann.get_output([1, 2, 5]))
    print("OUT:\n", ann.get_output([1, 3, 5]))
    print("OUT:\n", ann.get_output([1, 4, 5]))

