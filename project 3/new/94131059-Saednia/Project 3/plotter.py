__author__ = 'shiriiin'

import multiprocessing
import matplotlib.pyplot as plt
import numpy as np


class Plotter(multiprocessing.Process):
    def __init__(self, tasks_queue):
        multiprocessing.Process.__init__(self)
        self.daemon = True
        self.tasskQueue = tasks_queue

        self.figure = plt.figure()

        self.weightsPlot = self.figure.add_subplot(2, 1, 1)
        self.weightsPlot.axis([0, 60, -10, 10])
        self.weightLines = []

        self.trainPlot = self.figure.add_subplot(2, 1, 2)
        self.trainPlot.axis([0, 60, 0, 100])

        plt.grid()
        plt.ion()
        plt.show()

    def run(self):
        while 1:
            try:
                task = self.tasskQueue.get(timeout=.01)
            except Exception:
                plt.draw()
                plt.pause(.01)
                continue

            if task is None:
                self.tasskQueue.task_done()
                break

            # print task

            for k, v in task.iteritems():
                if k == 'weights':
                    if v.get('new', None):
                        for l in self.weightLines:
                            l.remove()
                        self.weightLines = []
                        for i in range(len(v['data'])):
                            self.weightLines.append(self.weightsPlot.plot(v['data'][i][0], v['data'][i][1])[0])
                    else:
                        for i in range(len(v['data'])):
                            self.weightLines[i].set_xdata(np.append(self.weightLines[i].get_xdata(), v['data'][i][0]))
                            self.weightLines[i].set_ydata(np.append(self.weightLines[i].get_ydata(), v['data'][i][1]))
                elif k == 'train':
                    if v.get('new', None):
                        self.trainLines = []
                        for i in range(len(v['data'])):
                            self.trainLines.append(self.trainPlot.plot(v['data'][i][0], v['data'][i][1])[0])
                    else:
                        print v['data']
                        for i in range(len(v['data'])):
                            self.trainLines[i].set_xdata(np.append(self.trainLines[i].get_xdata(), v['data'][i][0]))
                            self.trainLines[i].set_ydata(np.append(self.trainLines[i].get_ydata(), v['data'][i][1]))

            plt.draw()
            plt.pause(.2)

            self.tasskQueue.task_done()

        plt.ioff()
        plt.show()

        return
