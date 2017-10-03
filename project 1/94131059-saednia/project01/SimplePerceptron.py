__author__ = 'shiriiin'

import plotter
from random import shuffle
import numpy as np


class SimplePerceptron(object):
    def __init__(self, eta=0.1, epochs=50, error_threshold=0, weight=None, plot=None):
        self.eta = eta
        self.epochs = epochs
        self.error_threshold = error_threshold

        self.training_errors = []
        self.validation_errors = []

        self.input_training = []
        self.input_validation = []

        self.best_weight = []
        self.best_validation_errors = 1000

        if weight is not None:
            self.weight = weight

        self.is_finished = False
        self.plot = plot

    def get_data(self, input_file, training_percentage=0.7):
        data_set = []
        data_file = open(input_file, "r")
        for line in data_file:
            line = line.strip().split('\t')
            line = [float(x) for x in line]

            data_set.append([line[:-1], line[-1]])

        output_filter = lambda x: x[1]
        outputs_dimension = int(max(data_set, key=output_filter)[1])
        # print data_set
        # print outputs_dimension

        for data in data_set:
            tmp_l = [-1] * outputs_dimension
            if int(data[1]) != 0:
                tmp_l[int(data[1])-1] = 1
            data[1] = tmp_l

        print data_set
        shuffle(data_set)
        self.input_training = data_set[:(int(len(data_set) * training_percentage))]
        self.input_validation = data_set[(int(len(data_set) * training_percentage)):]

    def train(self, train_set):
        self.is_finished = False
        counter = 0

        if getattr(self, 'weight', None) is None:
            self.weight = np.random.randn(len(train_set[0][1]), 1 + len(train_set[0][0]))

        # plot section
        if len(train_set[0][1]) == 1:
            self.plot.refresh_sample_plotter([self.input_training, self.input_validation])
        self.plot.refresh_error_plotter(validation_error=self.test(train_set[:2]))
        self.plot.refresh_weights_plotter([w for weights in self.weight for w in weights])
        #

        for ep in range(self.epochs):
            errors = [0] * (len(self.weight)+1)
            for Xi, Yi in train_set:
                tmp_y = [0] * len(Yi)
                for j in range(len(self.weight)):
                    update = self.eta * (Yi[j] - self.predict(Xi, j))
                    self.weight[j][1:] += update * np.array(Xi)
                    self.weight[j][0] += update * 1
                    tmp_y[j] = self.predict(Xi, j)
                ind = Yi.index(1)+1 if 1 in Yi else 0
                errors[ind] += int(tmp_y != Yi)
                print tmp_y, Yi
            self.training_errors.append(errors)

            validation_error = self.test(self.input_validation)
            print "validation error: %s" % str(validation_error), "\t training error: %s" % str(errors)

            self.update_plot(len(train_set[0][0]), len(train_set[0][1]), validation_error, errors)

            if validation_error <= self.best_validation_errors:
                self.best_validation_errors = validation_error
                self.best_weight = self.weight
                counter = 0
            else:
                counter += 1

            # if counter >= 100:
            #     self.weight = self.best_weight
            #     print "break the loop because of over fitting at %d" % ep
            #     break

            if validation_error <= self.error_threshold:
                print "break the loop because of error satisfying at %d" % ep
                break

            # if sum(errors) == 0:
            #     print "break the loop because of training error 0 at %d" % ep
            #     break

        print "weights", self.weight
        self.is_finished = True

    def update_plot(self, input_dimensions, output_dimensions, validation_error, training_error):
        if output_dimensions == 1 and input_dimensions == 2:
            def line_formula(x, o):
                res = []
                m1 = -(self.weight[o][1]/self.weight[o][2])
                m0 = -(self.weight[o][0]/self.weight[o][2])
                for xi in x:
                    res.append(m1*xi+m0)
                return res
            self.plot.update_sample_plotter(x=[-100, 100], y=line_formula([-100, 100], 0))

        self.plot.update_error_plotter(validation_error, training_error)
        self.plot.update_weights_plotter([w for weights in self.weight for w in weights])

    def test(self, test_set):
        errors = [0] * (len(self.weight)+1)
        for Xi, Yi in test_set:
            tmp_y = [0] * len(Yi)
            for j in range(len(self.weight)):
                tmp_y[j] = self.predict(Xi, j)
            ind = Yi.index(1)+1 if 1 in Yi else 0
            errors[ind] += int(tmp_y != Yi)
        self.validation_errors.append(errors)
        return errors

    def net_input(self, x, j):
        return np.dot(x, self.weight[j][1:]) + self.weight[j][0]

    def predict(self, x, j):
        return np.where(self.net_input(x, j) >= 0.0, 1, -1)

    def save_weights(self, file_name):
        ff = open(file_name, 'w')
        for l in self.weight:
            for w in l:
                ff.write(str(w) + ' ')
            ff.write('\n')
        # print self.weight


if __name__ == '__main__':
    sp = SimplePerceptron(epochs=500, eta=0.01, error_threshold=-1)

    sp.get_data('Dataset1.data', 1)
    # sp.input_training = [([2, 2], [1]), ([2, 1], [-1])]

    sp.train(sp.input_training)

    # print"train errors"
    # print simple.errors
    # print simple.weight
    # simple.test(simple.input_validation)

    raw_input('finish?')