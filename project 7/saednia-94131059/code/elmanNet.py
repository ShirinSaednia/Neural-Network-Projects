
import multiprocessing
import numpy as np
from random import random
import plotter


def sigmoid(x):
    return np.tanh(x)


def dsigmoid(x):
    return 1.0-x**2


class Elman(multiprocessing.Process):
    def __init__(self, update_queue, *args):
        multiprocessing.Process.__init__(self)
        self.daemon = True
        self.update_queue = update_queue

        self.shape = args
        n = len(args)

        # Build layers
        self.layers = []

        # Input layer (+1 unit for bias
        #              +size of first hidden layer)
        self.layers.append(np.ones(self.shape[0]+1+self.shape[1]))

        # Hidden layer(s) + output layer
        for i in range(1, n):
            self.layers.append(np.ones(self.shape[i]))

        # Build weights matrix
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size,
                                          self.layers[i+1].size)))

        # dw will hold last change in weights (for momentum)
        self.dw = [0, ]*len(self.weights)

        # Reset weights
        self.reset()

    def run(self):
        self.train()

    def reset(self):
        for i in range(len(self.weights)):
            z = np.random.random((self.layers[i].size, self.layers[i+1].size))
            self.weights[i][...] = (2*z-1)*0.25

    def get_data(self, input_file, column):
        data_set = []
        data_file = open(input_file, "r")
        for line in data_file:
            line = line.strip().split(' ')
            line = [float(x) for x in line]
            data_set.append(np.array(line[:]))
        series = np.array(data_set).transpose()

        train_samples = np.zeros(10000, dtype=[('input',  float, 1), ('output', float, 1)])
        tmp = 0
        ind = 0
        for d in series[column][:10000]:
            train_samples[ind] = (tmp), (d)
            ind += 1
            tmp = d

        test_samples = np.zeros(10000, dtype=[('input',  float, 1), ('output', float, 1)])
        tmp = 0
        ind = 0
        for d in series[column][13000:23000]:
            test_samples[ind] = (tmp), (d)
            ind += 1
            tmp = d

        return train_samples, test_samples

    def propagate_forward(self, data):
        # Set input layer with data
        self.layers[0][:self.shape[0]] = data
        # and first hidden layer
        self.layers[0][self.shape[0]:-1] = self.layers[1]

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        for i in range(1, len(self.shape)):
            # Propagate activity
            self.layers[i][...] = sigmoid(np.dot(self.layers[i-1], self.weights[i-1]))

        # Return output
        return self.layers[-1]

    def propagate_backward(self, target, lrate=0.1, momentum=0.1):
        deltas = []

        # Compute error on output layer
        error = target - self.layers[-1]
        delta = error*dsigmoid(self.layers[-1])
        deltas.append(delta)

        # Compute error on hidden layers
        for i in range(len(self.shape)-2, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T)*dsigmoid(self.layers[i])
            deltas.insert(0, delta)

        # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T, delta)
            self.weights[i] += lrate*dw + momentum*self.dw[i]
            self.dw[i] = dw

        # Return error
        return (error**2).sum()

    def train(self):
        train_samples, test_samples = self.get_data("synthetic", 5)

        for ep in range(150):
            for j in range(train_samples.size):
                self.propagate_forward(train_samples['input'][j])
                self.propagate_backward(train_samples['output'][j], lrate=0.1, momentum=0)

            c = 0
            predicts = []
            for j in range(test_samples.size):
                o = self.propagate_forward(test_samples['input'][j])[0]
                c += (o - test_samples['output'][j])**2/2
                predicts.append(o)

            res = dict()
            res['error'] = {'data': c}
            res['series'] = {'data': predicts}
            res['weights'] = {'data': [(ep, w2[0]) for w1 in self.weights for w2 in w1]} #for w3 in w2]}
            if ep == 0:
                res['series']['main'] = test_samples['output'][:]
                res['weights']['new'] = True
            # print res['weights']
            # print self.weights

            print ep, '->', c

            self.update_queue.put(res)
            self.update_queue.join()


if __name__ == '__main__':
    updates_queue = multiprocessing.JoinableQueue()
    pl = plotter.Plotter(updates_queue)
    pl.start()

    network = Elman(updates_queue, 1, 4, 1)
    network.start()

    network.join()
    raw_input("finish?")