__author__ = 'shiriiin
'

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, max_plots=3):
        self.plots = 0
        self.max_plots = max_plots
        self.figure = plt.figure()

        self.plots += 1
        self.error_plotter_plot = self.figure.add_subplot(max_plots, 1, self.plots)
        self.error_plotter_plot.axis([0, 500, 0, 100])
        # self.error_plotter_plot.autoscale(True, axis='x', tight=False)

        self.plots += 1
        self.weights_plotter_plot = self.figure.add_subplot(max_plots, 1, self.plots)
        self.weights_plotter_plot.axis([0, 1000, -50, 50])
        # weights_plotter.plot.autoscale(True, axis='both', tight=False)

        self.plots += 1
        self.sample_plotter_plot = self.figure.add_subplot(max_plots, 1, self.plots)
        self.sample_plotter_plot.axis([-100, 100, -100, 100])
        # sample_plotter.plot.autoscale(True, axis='both', tight=False)

    def refresh_error_plotter(self, validation_error):
        self.error_plotter_iteration = 0
        self.error_plotter_validation_error_line = \
            [self.error_plotter_plot.plot([], [])[0] for i in range(len(validation_error))]
        self.error_plotter_training_error_line = \
            [self.error_plotter_plot.plot([], [])[0] for i in range(len(validation_error))]
        self.error_plotter_validation_error = [[] for i in range(len(validation_error))]
        self.error_plotter_training_error = [[] for i in range(len(validation_error))]

    def update_error_plotter(self, validation_error, training_error):
        self.error_plotter_iteration += 1
        tmp_xdata = range(self.error_plotter_iteration)
        for i in range(len(validation_error)):
            self.error_plotter_validation_error[i].append(validation_error[i])
            self.error_plotter_validation_error_line[i].set_xdata(tmp_xdata)
            self.error_plotter_validation_error_line[i].set_ydata(self.error_plotter_validation_error[i])

            self.error_plotter_training_error[i].append(training_error[i])
            self.error_plotter_training_error_line[i].set_xdata(tmp_xdata)
            self.error_plotter_training_error_line[i].set_ydata(self.error_plotter_training_error[i])

    def refresh_weights_plotter(self, weights):
        self.weights_plotter_iteration = 0
        self.weights_plotter_lines = [self.weights_plotter_plot.plot([], [])[0] for i in range(len(weights))]
        self.weights_plotter_weights = [[] for i in range(len(weights))]

    def update_weights_plotter(self, weights):
        self.weights_plotter_iteration += 1
        tmp_xdata = range(self.weights_plotter_iteration)
        for i in range(len(weights)):
            self.weights_plotter_weights[i].append(weights[i])
            self.weights_plotter_lines[i].set_xdata(tmp_xdata)
            self.weights_plotter_lines[i].set_ydata(self.weights_plotter_weights[i])

    def refresh_sample_plotter(self, samples=[]):
        self.sample_plotter_iteration = 0
        self.sample_plotter_line, = self.sample_plotter_plot.plot([], [])

        for sample in samples:
            classes = {}
            for s in sample:
                tmp_key = int(s[1][0] == 1)
                if not tmp_key in classes:
                    classes[tmp_key] = []
                classes[tmp_key].append(s[0])

            c1x, c1y = [], []
            c2x, c2y = [], []
            for k, v in classes.iteritems():
                if k == 1:
                    for ind in v:
                        c1x.append(ind[0])
                        c1y.append(ind[1])
                else:
                    for ind in v:
                        c2x.append(ind[0])
                        c2y.append(ind[1])
            self.sample_plotter_plot.plot(c1x, c1y, 'r.')
            self.sample_plotter_plot.plot(c2x, c2y, 'b.')

    def update_sample_plotter(self, x, y):
        self.sample_plotter_iteration += 1
        self.sample_plotter_line.set_xdata(x)
        self.sample_plotter_line.set_ydata(y)


def plot_draw(sl=0.01):
    plt.draw()
    plt.pause(sl)

if __name__ == "__main__":
    samples = []