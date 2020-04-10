import numpy as np
import matplotlib.pyplot

# neural network class definition
class Ann:
    def __init__(self, i_nodes, h_nodes, o_nodes, learning_rate):
        # Number of nodes per layer
        self.I = i_nodes
        self.J = h_nodes
        self.K = o_nodes
        
        # Weight matrices
        self.wij = np.random.rand(self.J, self.I)
        self.wjk = np.random.rand(self.K, self.J)

        # learning rate
        self.lr = learning_rate
        
        # activation function is the sigmoid function
        self.activation = self._sigmoid

    def _sigmoid(self, x):
       return 1.0 / (1.0 + np.e ** (-x))
    
    def train(self, input_vector, target_vector):
        # convert received lists into 2d arrays
        inputs = np.array(input_vector, ndmin=2).T
        # print("X:\n", inputs)
        targets = np.array(target_vector, ndmin=2).T
        # print("T:\n", targets)
        
        # calculate internal states and activation of hidden neurons
        hidden_states = np.dot(self.wij, inputs)
        # print("HS\n", hidden_states)
        hidden_activations = self.activation(hidden_states)
        # print("Y\n", hidden_activation)
        
        # calculate internal states and activation of output neurons
        final_states = np.dot(self.wjk, hidden_activations)
        # calculate the signals emerging from final output layer
        output_activations = self.activation(final_states)
        
        # Output layer errors
        output_errors = output_activations - targets
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


    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wij, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.wjk, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation(final_inputs)
        
        return final_outputs


if __name__ == "__main__":
    # number of input, hidden and output nodes
    i_nodes = 784
    h_nodes = 200
    o_nodes = 10

    # learning rate
    learning_rate = 0.35

    # create instance of neural network
    ann = Ann(i_nodes, h_nodes,o_nodes, learning_rate)

    # load the mnist training data CSV file into a list
    training_data_file = open("data/mnist_2_train_60000_.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    size = len(training_data_list)

    # load the mnist test data CSV file into a list
    test_data_file = open("data/mnist_2_test_10000_.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # train the neural network
    print("TRAINING")
    # An epoch is one run through the training data set
    epochs = 5

    i = 0
    for e in range(epochs):
        # go through all records in the training data set
        for record in training_data_list:
            i += 1
            if i % 1000 == 0:
                print( format(i / (epochs * size), '8.5f'))
            # split the record by the ',' commas
            all_values = record.split(',')
            # scale and shift the inputs
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # create the target output values (all 0.01, except the desired label which is 0.99)
            targets = np.zeros(o_nodes) + 0.01
            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            ann.train(inputs, targets)
    i = 0

    # test the neural network
    print("TESTING")
    # scorecard for how well the network performs
    scorecard = []

    # go through all the records in the test data set
    for record in test_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # correct answer is first value
        correct_label = int(all_values[0])
        # scale and shift the inputs
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = ann.query(inputs)
        # the index of the highest value corresponds to the label
        label = np.argmax(outputs)
        # append correct or incorrect to list
        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)

    # calculate the performance score, the fraction of correct answers
    scorecard_array = np.asarray(scorecard)
    print ("performance = ", scorecard_array.sum() / scorecard_array.size)

