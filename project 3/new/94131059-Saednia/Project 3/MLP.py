__author__ = 'shiriiin'

import random
import multiprocessing
from copy import deepcopy
import numpy as np
import readInput
import sys

training_data, validation_data = readInput.readData(False, 1)


class Network(multiprocessing.Process):
    def __init__(self, sizes, epochs, eta, mu, update_queue, validation_threshold, error_threshold,
                 widrow=False, mini_batch_size=10, activation_function='linear'):

        multiprocessing.Process.__init__(self)
        self.daemon = True
        self.updateQueue = update_queue

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.epochs = epochs
        self.eta = eta
        self.mu = mu
        self.mini_batch_size = mini_batch_size
        self.validation_threshold = validation_threshold
        self.error_threshold = error_threshold
        self.activation_function = activation_function

        if widrow:
            scaleFactor = sizes[1]**(1.0/sizes[0])
            self.biases = [np.random.uniform(-scaleFactor, scaleFactor, (y, 1)) for y in sizes[1:]]
            self.weights = [np.random.uniform(-0.5, 0.5, (y, x)) for x, y in zip(sizes[:-1], sizes[1:])]
            #print self.weights
            for weight in self.weights:
                for wj in weight:
                    norm = np.linalg.norm(wj)
                    for wi in wj:
                        wi = scaleFactor*wi/norm
        else:
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
            self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases_history = [[np.zeros((y, 1)) for y in sizes[1:]]]*2
        self.weights_history = [[np.zeros((y, x)) for x, y in zip(sizes[:-1], sizes[1:])]]*2

    def run(self):
        global training_data, validation_data, test_data
        self.SGD(training_data, self.epochs, self.mini_batch_size, self.eta, self.mu, validation_data, self.validation_threshold, self.error_threshold)
        print "finishhhh"
        sys.exit(0)
        #self.SGD2(training_data, evaluate_training, self.epochs, 10, self.eta, validation_data, self.validation_threshold, self.error_threshold)

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):

            if self.activation_function == "linear":
                a = linear(np.dot(w, a)+b)

            elif self.activation_function == "piecewiselinear":
                a = piecewiselinear(np.dot(w, a)+b)

            elif self.activation_function == "logarithm":
                a = logarithm(np.dot(w, a)+b)

        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, mu, validation_data=None, counter_thresh=None, error_thresh=None):
        if validation_data:
            best_result_index = 0
            counter_validation = 0
            mem_validation = []
        n = len(training_data)

        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            bprim = [mu*(b1-b0) for b0, b1 in zip(self.biases_history[0], self.biases_history[1])]
            self.biases = [b+bd for b, bd in zip(self.biases, bprim)]
            self.biases_history.pop(0)
            self.biases_history.append(deepcopy(self.biases))

            wprim = [mu*(w1-w0) for w0, w1 in zip(self.weights_history[0], self.weights_history[1])]
            self.weights = [w+wd for w, wd in zip(self.weights, wprim)]
            self.weights_history.pop(0)
            self.weights_history.append(deepcopy(self.weights))

            res = dict()
            res['weights'] = {'data': [(j, w3) for w1 in self.weights for w2 in w1 for w3 in w2]}
            res['train'] = {'data': [(j, self.evaluate(training_data)/n)]}
            if j == 0:
                res['weights']['new'] = True
                res['train']['new'] = True

            if validation_data:
                ev = self.evaluate(validation_data)
                mem_validation.append(ev)

                res['train']['data'].append((j, ev/len(validation_data)))

                if mem_validation[-1] >= mem_validation[best_result_index]:
                    best_result_index = len(mem_validation) - 1
                    counter_validation = 0
                else:
                    counter_validation += 1

                if counter_thresh and counter_thresh == counter_validation:
                    print "Epoch {0}: OverFitting Detected".format(j)
                    break

                if error_thresh and error_thresh >= mem_validation[-1]:
                    print "Epoch {0}: Error threshold satisfied.".format(j)
                    break

                print "Epoch {0}: {1}".format(j, ev)
            else:
                print "Epoch {0} complete".format(j)

            self.updateQueue.put(res)
            self.updateQueue.join()

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            if self.activation_function == "linear":
                activation = linear(z)

            elif self.activation_function == "piecewiselinear":
                activation = piecewiselinear(z)

            elif self.activation_function == "logarithm":
                activation = logarithm(z)

            #activation = linear(z)
            activations.append(activation)
        # backward pass
        if self.activation_function == "linear":
            delta = self.cost_derivative(activations[-1], y) * linear_prime(zs[-1])

        elif self.activation_function == "piecewiselinear":
            delta = self.cost_derivative(activations[-1], y) * piecewiselinear_prime(zs[-1])

        elif self.activation_function == "logarithm":
            delta = self.cost_derivative(activations[-1], y) * logarithm_prime(zs[-1])

        #delta = self.cost_derivative(activations[-1], y) * linear_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            if self.activation_function == "linear":
                sp = linear_prime(z)

            elif self.activation_function == "piecewiselinear":
                sp = piecewiselinear_prime(z)

            elif self.activation_function == "logarithm":
                sp = logarithm_prime(z)

            #sp = linear_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(self.feedforward(x)[0][0], y[0][0]) for (x, y) in test_data]
        # print test_results
        return sum(np.absolute(x-y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)


def linear(z):
    return z


def linear_prime(z):
    return 1


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def piecewiselinear(z):
    return np.where(z < 0, 0, np.where(z > 180, 180, z))


def piecewiselinear_prime(z):
    return np.where(z < 0, 0, np.where(z > 180, 0, 1))


def logarithm(z):
    return np.where(z > 0.1, np.log10(z), 0)


def logarithm_prime(z):
    return 1.0/(z*np.log(10))