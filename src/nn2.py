import math
import random

class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_layers):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.weights = self.initialize_weights()
        self.biases = self.initialize_biases()

    def initialize_weights(self):
        weights = []
        for i in range(len(self.hidden_layers) + 1):
            if i == 0:
                # Input layer to first hidden layer
                new_weights = [[random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)] for _ in range(self.input_size)]
            else:
                # Hidden layer to next hidden layer
                new_weights = [[random.uniform(-0.1, 0.1) for _ in range(self.hidden_layers[i-1])] for _ in range(self.hidden_layers[i-1])]
            
            weights.append(new_weights)
        return weights

    def initialize_biases(self):
        biases = []
        for i in range(len(self.hidden_layers)):
            new_biases = [random.uniform(-0.1, 0.1) for _ in range(self.hidden_layers[i])]
            biases.append(new_biases)
        
        # Output layer
        if self.output_size > 1:
            new_biases = [random.uniform(-0.1, 0.1) for _ in range(2)]
            biases.append(new_biases)
        return biases

    def sigmoid(self, x):
        if not isinstance(x, list): 
            return 1 / (1 + math.exp(-x))
        else:
            return [1 / (1 + math.exp(-i)) for i in x]
    def softmax(self, x):
        e_x = [math.exp(i) for i in x]
        return [i/e_x[0] for i in e_x]

    def forward_propagation(self, inputs):
        output = self.sigmoid(inputs)

        # Hidden layers
        for layer in range(len(self.hidden_layers)):
            hidden_layer_values = [[0 for _ in range(self.input_size)] for _ in range(self.hidden_layers[layer])]
            
            # Input to hidden layer
            if layer == 0:
                new_values = [input * weight for input, weight in zip(inputs, self.weights[0])]
            else:
                new_values = [self.sigmoid(sum(input * weight for input, weight in zip(hidden_layer_values[i], self.weights[layer])) + bias) for i, bias in enumerate(self.biases[layer])]

            hidden_layer_values = new_values
            output = hidden_layer_values

        return output


if __name__ == "__main__":
    # Create a neural network with 2 inputs and 1 output
    nn = NeuralNetwork(2, 1, [3])

    # Initialize weights and biases
    print("Weights:")
    for layer in nn.weights:
        for sublayer in layer:
            print(sublayer)

    print("\nBiases:")
    for i, bias in enumerate(nn.biases):
        print(f"Hidden Layer {i+1}: {bias}")

    # Create inputs
    inputs = [0.5, 0.3]

    # Perform forward propagation
    output = nn.forward_propagation(inputs)
    print("Output:", output)

    # Sigmoid activation function
    print("\nSigmoid Activation Function:")
    for i in range(len(nn.weights)):
        hidden_layer_values = [[0 for _ in range(2)] for _ in range(3)]
        new_values = [input * weight for input, weight in zip(inputs, nn.weights[i])]
        sigmoid_output = self.sigmoid(sum(new_values) + nn.biases[i][0])
        
        print(f"Hidden Layer {i+1}:")
        for j, value in enumerate(hidden_layer_values):
            hidden_layer_values[j] = sigmoid_output
        print(hidden_layer_values)