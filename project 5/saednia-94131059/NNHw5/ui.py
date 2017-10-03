__author__ = 'shiriiin'

import CNN
import matplotlib
matplotlib.use('GTKAgg')

from plotter import Plotter
import multiprocessing
from Tkinter import *

class CNNUi(Tk):
    def __init__(self):
        Tk.__init__(self)

        self.updatesQueue = multiprocessing.JoinableQueue()
        self.plotter = Plotter(self.updatesQueue)
        self.cnn = None

        self.title("HW5")

        self.mini_batch_size = Entry(self)
        self.trainingPercent = Entry(self)
        self.epochs = Entry(self)
        self.validation_threshold = Entry(self)
        self.error_threshold = Entry(self)
        self.networkAddress = Entry(self)

        self.poolingRule = StringVar()

    def add_entry(self):
        Label(self, text="Add Parameter for RBF learning").grid(row=0)

        Label(self, text="MiniBatchSize").grid(row=2)
        Label(self, text="TrainingPercent").grid(row=3)
        Label(self, text="Epochs").grid(row=4)
        Label(self, text="ValidationThreshold").grid(row=5)
        Label(self, text="ErrorThreshold").grid(row=6)
        Label(self, text="NetworkAdress").grid(row=7)
        Label(self, text="PoolingRule").grid(row=9)

        Button(self, text="Train", command=self.train).grid(row=15, column=0, sticky=W, pady=4)
        # Button(self, text="Save Weights", command=self.save_weights).grid(row=11, column=1, sticky=W, pady=4)

        self.mini_batch_size.grid(row=2, column=1)
        self.trainingPercent.grid(row=3, column=1)
        self.epochs.grid(row=4, column=1)
        self.validation_threshold.grid(row=5, column=1)
        self.error_threshold.grid(row=6, column=1)
        self.networkAddress.grid(row=7, column=1)

        self.mini_batch_size.insert(0, "10")
        self.trainingPercent.insert(0, "1")
        self.epochs.insert(0, 30)
        self.validation_threshold.insert(0, 30)
        self.error_threshold.insert(0, -1)

        Radiobutton(self, text="Max", padx=20, variable=self.poolingRule, value="max").grid(row=11, column=0, sticky=W, pady=4)
        Radiobutton(self, text="Mean", padx=20, variable=self.poolingRule, value="mean").grid(row=12, column=0, sticky=W, pady=4)


    def train(self):
        # self.trainBuntton["state"] = 'disabled'
                #sizes, epochs, eta, update_queue, validation_threshold, error_threshold
        trainingPercent = int(self.trainingPercent.get())
        epochs = int(self.epochs.get())
        et = float(self.error_threshold.get())
        vt = int(self.validation_threshold.get())
        poolinRule = self.poolingRule.get()
        mini_batch_size = int(self.mini_batch_size.get())

        try:
            import pickle
            weights = pickle.load(open(self.address.get(), "rb"))
        except:
            weights = None

        if not self.plotter.is_alive():
            self.plotter.start()

        if self.cnn is not None:
            if self.cnn.exitcode != 0:
                print 'Another learner is working.'
                return
            self.cnn.terminate()

        # # self.cnn = CNN.Network(sizes, epochs, eta, mu, self.updatesQueue, vt, et, widrow, bs, activationFunc, weights)
        # self.cnn = CNN.Network([CNN.ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), filter_shape=(20, 1, 5, 5), poolsize=(2, 2)),
        #                         CNN.FullyConnectedLayer(n_in=20*12*12, n_out=100),
        #                         CNN.SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size, self.updatesQueue)

        self.cnn = CNN.Network([
                CNN.ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                              filter_shape=(20, 1, 5, 5),
                              poolsize=(2, 2)),
                CNN.ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                              filter_shape=(30, 20, 5, 5),
                              poolsize=(2, 2)),
                CNN.ConvPoolLayer(image_shape=(mini_batch_size, 30, 4, 4),
                              filter_shape=(40, 30, 2, 2),
                              poolsize=(3, 3)),
                CNN.FullyConnectedLayer(n_in=40*1*1, n_out=100),
                CNN.SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size, self.updatesQueue)
        self.cnn.start()

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
    w = CNNUi()
    w.add_entry()
    w.mainloop()