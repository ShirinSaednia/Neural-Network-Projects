__author__ = 'shiriiin'

from Tkinter import *
import threading

import matplotlib
matplotlib.use('GTKAgg')

import matplotlib.pyplot as plt

import plotter
import SimplePerceptron
import Adaline
import SecondOrderPerceptron


class PerceptronUi(Tk):
    def __init__(self):
        Tk.__init__(self)
        
        self.nn = None
        self.tt = None
        self.plot = plotter.Plotter()
        plt.ion()  # make show non-blocking
        plt.show()  # show the figure

        self.title("HW1")
        self.e1 = Entry(self)
        self.e2 = Entry(self)
        self.e3 = Entry(self)
        self.e4 = Entry(self)
        self.e5 = Entry(self)
        self.r1 = StringVar()
        self.r2 = IntVar()

        # self.text_pad()

    def add_entry(self):
        #Label(self, text="Add Parameter for perceptron learning").grid(row=0)
        Label(self, text="Alfa").grid(row=3)
        Label(self, text="Epochs").grid(row=4)
        Label(self, text="Train Percent Samples").grid(row=5)
        Label(self, text="Error Threshold").grid(row=6)
        Label(self, text="Weights Address").grid(row=7)

        Button(self, text="Train", command=self.train).grid(row=11, column=0, sticky=W, pady=4)
        Button(self, text="Save Weights", command=self.save_weights).grid(row=11, column=1, sticky=W, pady=4)
        # Button(self, text="Quit", command=self.quit).grid(row=12, column=1, sticky=W, pady=4)

        self.e1.grid(row=3, column=1)
        self.e2.grid(row=4, column=1)
        self.e3.grid(row=5, column=1)
        self.e4.grid(row=6, column=1)
        self.e5.grid(row=7, column=1)

        self.e1.insert(10, 0.01)
        self.e2.insert(10, 100)
        self.e3.insert(10, 1)
        self.e4.insert(10, -1)

        Radiobutton(self, text="Dataset1", padx=20, variable=self.r1, value="Dataset1.data").grid(row=8, column=0, sticky=W, pady=4)
        Radiobutton(self, text="Dataset2", padx=20, variable=self.r1, value="Dataset2.data").grid(row=9, column=0, sticky=W, pady=4)
        Radiobutton(self, text="Dataset3", padx=20, variable=self.r1, value="Dataset3.data").grid(row=10, column=0, sticky=W, pady=4)

        Radiobutton(self, text="Perceptron", padx=20, variable=self.r2, value=1).grid(row=0, column=0, sticky=W, pady=4)
        Radiobutton(self, text="Adaline", padx=20, variable=self.r2, value=2).grid(row=1, column=0, sticky=W, pady=4)
        Radiobutton(self, text="Second Order Perceptron", padx=20, variable=self.r2, value=3).grid(row=2, column=0, sticky=W, pady=4)

    def train(self):
        if self.nn is not None and self.nn.is_finished is not True:
            print "--- already working"
            return

        eta = float(self.e1.get())
        epochs = int(self.e2.get())
        percent = float(self.e3.get())
        threshold = int(self.e4.get())
        weight = str(self.e5.get())
        data_file = self.r1.get()
        learning_algorithm = self.r2.get()

        try:
            ff = open(weight, 'r')
            weight = []
            for l in ff:
                l = l.strip().split()
                l = [float(i) for i in l]
                weight.append(l)
            weight = SimplePerceptron.np.array(weight)
        except:
            weight = None

        if learning_algorithm == 1:
            self.nn = SimplePerceptron.SimplePerceptron(eta, epochs, threshold, weight, self.plot)
        elif learning_algorithm == 2:
            self.nn = Adaline.Adaline(eta, epochs, threshold, weight, self.plot)
        elif learning_algorithm == 3:
            self.nn = SecondOrderPerceptron.SecondOrderPerceptron(eta, epochs, threshold, weight, self.plot)
        else:
            return

        self.tt = threading.Thread(target=self._worker, args=(data_file, percent,))
        self.tt.setDaemon(True)
        self.tt.start()

    def _worker(self, data_file, percent):
        self.nn.get_data(data_file, percent)
        self.nn.train(self.nn.input_training)

    def save_weights(self):
        if self.nn is None:
            return
        self.nn.save_weights('saved_weights')


from time import sleep
if __name__ == '__main__':
    w = PerceptronUi()
    w.add_entry()
    # w.mainloop()
    try:
        while True:
            w.update_idletasks()
            w.update()
            plotter.plot_draw(0.01)
    except Exception:
        pass
