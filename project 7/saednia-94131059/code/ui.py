__author__ = 'shiriiin'

import matplotlib
matplotlib.use('GTKAgg')
import elmanNet


from plotter import Plotter
import multiprocessing
from Tkinter import *


class elmanUi(Tk):
    def __init__(self):
        Tk.__init__(self)

        self.updatesQueue = multiprocessing.JoinableQueue()
        self.plotter = Plotter(self.updatesQueue)
        self.elman = None

        self.title("HW7")

        self.arch = Entry(self)
        self.backRate = Entry(self)
        self.learningRate = Entry(self)
        self.batchSize = Entry(self)
        self.seriesNum = Entry(self)
        self.stepNum = Entry(self)
        self.networkAddress = Entry(self)

        self.errorFunc = StringVar()

    def add_entry(self):
        Label(self, text="Add Parameter for Elman learning").grid(row=0)
        Label(self, text="Architecture").grid(row=3)
        Label(self, text="BakhpropagationRates").grid(row=4)
        Label(self, text="LearningRates").grid(row=5)
        Label(self, text="BatchSize").grid(row=6)
        Label(self, text="SeriesNum").grid(row=7)
        Label(self, text="StepsNum").grid(row=8)
        Label(self, text="NetworkAdress").grid(row=9)

        Button(self, text="Train", command=self.train).grid(row=19, column=0, sticky=W, pady=4)
        # Button(self, text="Save Weights", command=self.save_weights).grid(row=11, column=1, sticky=W, pady=4)

        self.arch.grid(row=3, column=1)
        self.backRate.grid(row=4, column=1)
        self.learningRate.grid(row=5, column=1)
        self.batchSize.grid(row=6, column=1)
        self.seriesNum.grid(row=7, column=1)
        self.stepNum.grid(row=8, column=1)
        self.networkAddress.grid(row=9, column=1)


        self.arch.insert(0, "1,1,1")
        self.backRate.insert(0, 1)
        self.learningRate.insert(0, 0.01)
        self.batchSize.insert(0, 1)
        self.seriesNum.insert(0, 1)
        self.stepNum.insert(0, 1)

        Radiobutton(self, text="MSE", padx=20, variable=self.errorFunc, value="MSE").grid(row=11, column=0, sticky=W, pady=4)
        Radiobutton(self, text="MAE", padx=20, variable=self.errorFunc, value="MAE").grid(row=12, column=0, sticky=W, pady=4)
        Radiobutton(self, text="RMAE", padx=20, variable=self.errorFunc, value="RMAE").grid(row=13, column=0, sticky=W, pady=4)
        Radiobutton(self, text="PI", padx=20, variable=self.errorFunc, value="PI").grid(row=14, column=0, sticky=W, pady=4)

    def train(self):

        arch = [int(c) for c in str(self.arch.get()).strip().split(',')]
        backR = float(self.backRate.get())
        learnR = float(self.learningRate.get())
        bs = int(self.batchSize.get())
        serie = int(self.seriesNum.get())
        step = int(self.stepNum.get())
        errFunc = self.errorFunc.get()

        if not self.plotter.is_alive():
            self.plotter.start()

        if self.elman is not None:
            if self.elman.exitcode != 0:
                print 'Another learner is working.'
                return
            self.elman.terminate()

        self.elman = elmanNet.Elman(self.updatesQueue,1,1,1)
        self.elman.start()


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
    w = elmanUi()
    w.add_entry()
    w.mainloop()