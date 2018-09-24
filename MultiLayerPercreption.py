import math
import random
import pandas as pd
from matplotlib import pyplot as plt

class Data:
        pass

class Classifier:

    def __init__(self):
        self.__train = Data()
        self.__weights = []
        self.__biases = []
        self.__predictions = []
        self.__errors = []
    
    def train(self, train_data, train_fact, epoch=100, learning_rate=0.1, hidden_layer=[0], output_layer=1):
        self.__train.data = train_data
        self.__train.fact = train_fact
        self.__data_sample = self.__train.data[0]
        self.__epoch = epoch
        self.__neurons = hidden_layer + [output_layer]
        self.__layers = len(self.__neurons)
        self.__learning_rate = learning_rate
        self.__init_weights(hidden_layer, output_layer)
        self.__training()
        print('Training Finish')

    def plot(self):
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.xlim(1, len(self.__errors))
        plt.ylim(0, max(self.__errors))
        plt.plot(range(1,len(self.__errors) + 1), self.__errors)
        plt.grid(True)
        plt.show()

    def get_errors(self):
        return self.__errors

    def __training(self):
        for _ in range(self.__epoch):
            error = 0
            for i in range(len(self.__train.data)):
                self.__feedforward(self.__train.data[i])
                self.__backpropagation(self.__train.data[i], self.__train.fact[i])
                error = self.__loss_function(self.__predictions[len(self.__neurons) - 1][0], self.__train.fact[i])
            self.__errors.append(error)

    def __feedforward(self, data):
        for layer in range(self.__layers):
            for neuron in range(self.__neurons[layer]):
                data_used = data if layer == 0 else self.__predictions[layer - 1]
                target = self.__target_function(data_used, self.__weights[layer][neuron], self.__biases[layer][neuron])
                self.__predictions[layer][neuron] = self.__activation_function(target)

    def __backpropagation(self, data, fact):
        tau = [[] for _ in range(self.__layers)]
        for layer in range(self.__layers - 1, -1, -1):
            for neuron in range(self.__neurons[layer]):
                prediction = self.__predictions[layer][neuron]
                # Output to Hidden Layer
                if layer == self.__layers - 1:
                    tau[layer].append(self.__tau_function(prediction, fact=fact))
                # Hidden Layer to Hidden/Input Layer
                else:
                    total = 0.0
                    for next_neuron in range(self.__neurons[layer + 1]):
                        total += tau[layer + 1][next_neuron] * self.__weights[layer + 1][next_neuron][neuron]
                    tau[layer].append(self.__tau_function(prediction, total=total))
                data_used = data if layer == 0 else self.__predictions[layer - 1]
                # Update Weights
                for i, data_prev in enumerate(data_used):
                    self.__weights[layer][neuron][i] -= self.__learning_rate * self.__delta_function(tau[layer][neuron], data_prev)
                # Update Bias
                    self.__biases[layer][neuron] -= self.__learning_rate * self.__delta_function(tau[layer][neuron], 1)

    def __delta_function(self, tau, data):
        return tau * data    

    def __tau_function(self, prediction, fact=None, total=None):
        if total == None:
            return (fact - prediction) * (1 - prediction) * prediction
        else:
            return total * (1 - prediction) * prediction

    def __activation_function(self, target):
        # Sigmoid
        return 1 / (1 + math.exp(-target))

    def __target_function(self, data, weights, bias):
        total = 0.0
        for i in range(len(data)):
            total += data[i] * weights[i]
        total += bias
        return total

    def __loss_function(self, prediction, fact):
        return ((prediction - fact) ** 2) / 2

    def __init_weights(self, hidden_layer, output_layer):
        for layer in range(len(hidden_layer)):
            self.__weights.append([])
            self.__biases.append([])
            self.__predictions.append([])
            for _ in range(hidden_layer[layer]):
                if layer == 0:
                    self.__weights[layer].append([random.random() for _ in range(len(self.__train.data[0]))])
                else:
                    self.__weights[layer].append([random.random() for _ in range(hidden_layer[layer - 1])])
                self.__biases[layer].append(random.random())
                self.__predictions[layer].append(0)
        # Output Layer
        self.__weights.append([])
        self.__biases.append([])
        self.__predictions.append([])
        for _ in range(output_layer):
            self.__weights[len(hidden_layer)].append([random.random() for _ in range(hidden_layer[len(hidden_layer) - 1])])
            self.__biases[len(hidden_layer)].append(random.random())
            self.__predictions[len(hidden_layer)].append(0)

def read_data(file_path):
    try:
        data = []
        fact = []
        file = open(file_path)
        for i, line in enumerate(file):
            data.append([])
            attrs = line.strip().split(',')
            for col, attr in enumerate(attrs):
                if col < (len(attrs) - 1):
                    data[i].append(float(attr))
                else:
                    fact.append(int(attr))
    finally:
        file.close()
        return data, fact

def main():
    # Read data
    train = Data()
    file_path = 'Banknote_authentication.csv'
    train.data, train.fact = read_data(file_path)
    # Classifier
    clf = Classifier()
    clf.train(train.data, train.fact, epoch=50, learning_rate=0.85, hidden_layer=[2, 3, 2])
    print(clf.get_errors())
    clf.plot()

if __name__ == "__main__":
    main()
