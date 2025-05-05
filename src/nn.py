import math
import random

class Matrix

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        # Initialize weights randomly between -1 and 1
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.weights_ih = [random.random() * 2 - 1 for _ in range(input_nodes * hidden_nodes)]
        self.weights_ho = [random.random() * 2 - 1 for _ in range(hidden_nodes * output_nodes)]
        self
        
    def sigmoid(self, x):
        #     1
        # ---------
        # 1 + e^(-x)
        return 1 / (1 + math.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward_propagations(self, inputs):
        self.layers = []
        

# input_node 1 --w1-- hidden_node 1  __w3__ output_node 1
#              \_w2__ hidden_node 2 __w4_/

# input nodes take in input variables