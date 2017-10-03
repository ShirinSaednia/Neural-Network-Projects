__author__ = 'shiriiin'

from Tkinter import *
import threading

import matplotlib
matplotlib.use('GTKAgg')

import matplotlib.pyplot as plt

import MLP
from plotter import Plotter
import multiprocessing

class PerceptronUi(Tk):
    def __init__(self):
        Tk.__init__(self)

        self.updatesQueue = multiprocessing.JoinableQueue()
        self.plotter = Plotter(self.updatesQueue)
        self.mlp = None

        self.title("HW3")

        self.sizes = Entry(self)
        self.epochs = Entry(self)
        self.eta = Entry(self)
        self.validation_threshold = Entry(self)
        self.error_threshold = Entry(self)
        self.mu = Entry(self)
        self.batch_size = Entry(self)

        self.activationFunc = StringVar()
        self.widrow = StringVar()

    def add_entry(self):
        Label(self, text="Add Parameter for MLP learning").grid(row=0)
        Label(self, text="Size").grid(row=3)
        Label(self, text="Epochs").grid(row=4)
        Label(self, text="Eta").grid(row=5)
        Label(self, text="ValidationThreshold").grid(row=6)
        Label(self, text="ErrorThreshold").grid(row=7)
        Label(self, text="Mu").grid(row=8)
        Label(self, text="BatchSize").grid(row=9)
        Label(self, text="WidrowRule").grid(row=14)

        Button(self, text="Train", command=self.train).grid(row=17, column=0, sticky=W, pady=4)
        # Button(self, text="Save Weights", command=self.save_weights).grid(row=11, column=1, sticky=W, pady=4)

        self.sizes.grid(row=3, column=1)
        self.epochs.grid(row=4, column=1)
        self.eta.grid(row=5, column=1)
        self.validation_threshold.grid(row=6, column=1)
        self.error_threshold.grid(row=7, column=1)
        self.mu.grid(row=8, column=1)
        self.batch_size.grid(row=9, column=1)


        self.sizes.insert(0, "384, 20, 1")
        self.epochs.insert(0, 30)
        self.eta.insert(0, 0.003)
        self.validation_threshold.insert(0, 30)
        self.error_threshold.insert(0, -1)
        self.mu.insert(0, 0.01)
        self.batch_size.insert(0, 10)

        Radiobutton(self, text="Linear", padx=20, variable=self.activationFunc, value="linear").grid(row=10, column=0, sticky=W, pady=4)
        Radiobutton(self, text="PieceWiseLinear", padx=20, variable=self.activationFunc, value="piecewiselinear").grid(row=11, column=0, sticky=W, pady=4)
        Radiobutton(self, text="Logarithm", padx=20, variable=self.activationFunc, value="logarithm").grid(row=12, column=0, sticky=W, pady=4)

        Radiobutton(self, text="True", padx=20, variable=self.widrow, value="True").grid(row=15, column=0, sticky=W, pady=4)
        Radiobutton(self, text="False ", padx=20, variable=self.widrow, value="False").grid(row=16, column=0, sticky=W, pady=4)


    def train(self):
        # self.trainBuntton["state"] = 'disabled'
                #sizes, epochs, eta, update_queue, validation_threshold, error_threshold
        sizes = [int(i) for i in self.sizes.get().split(',')]
        eta = float(self.eta.get())
        epochs = int(self.epochs.get())
        et = float(self.error_threshold.get())
        vt = int(self.validation_threshold.get())
        mu = float(self.mu.get())
        bs = int(self.batch_size.get())
        activationFunc = self.activationFunc.get()
        widrow = self.widrow.get()

        if not self.plotter.is_alive():
            self.plotter.start()

        if self.mlp is not None:
            if self.mlp.exitcode != 0:
                print 'Another learner is working.'
                return
            self.mlp.terminate()

        self.mlp = MLP.Network(sizes, epochs, eta, mu, self.updatesQueue, vt, et, widrow, bs, activationFunc)
        self.mlp.start()


    # def _worker(self, data_file, percent):
    #     self.nn.get_data(data_file, percent)
    #     self.nn.train(self.nn.input_training)
    #
    # def save_weights(self):
    #     if self.nn is None:
    #         return
    #     self.nn.save_weights('saved_weights')


from time import sleep
if __name__ == '__main__':
    w = PerceptronUi()
    w.add_entry()
    w.mainloop()