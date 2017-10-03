__author__ = 'shiriiin'

import multiprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np


class Plotter(multiprocessing.Process):
    def __init__(self, tasks_queue):
        multiprocessing.Process.__init__(self)
        self.daemon = True
        self.tasskQueue = tasks_queue

        self.figure = plt.figure()

        self.weightsPlot = self.figure.add_subplot(2, 1, 1, projection='3d')
        # self.weightsPlot.axis([0, 60, -10, 10])
        self.weightsPlot.set_zlim(-3, 3)
        self.weightLine = None
        self.weightsData = None

        self.trainPlot = self.figure.add_subplot(2, 1, 2)
        self.trainPlot.axis([0, 30, 0, 1])

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
                if k == 'finish':
                    p = self.plot_filters(v[0], v[1], v[2])
                    plt.savefig("net_full_layer_0.png")
                elif k == 'weights':
                    if v.get('new', None):
                        if self.weightLine is not None:
                            self.weightLine.remove()
                        self.weightsData =[]
                        self.weightsData.append([v['data'][i][1] for i in range(len(v['data']))])
                        X, Y = np.meshgrid(range(len(v['data'])), range(v['data'][0][0]+1))
                        self.weightLine = self.weightsPlot.plot_surface(X, Y, self.weightsData, rstride=1, cstride=1, cmap=cm.coolwarm)
                    else:
                        self.weightLine.remove()
                        self.weightsData.append([v['data'][i][1] for i in range(len(v['data']))])
                        X, Y = np.meshgrid(range(len(v['data'])), range(v['data'][0][0]+1))
                        # print X, Y, self.weightsData
                        self.weightLine = self.weightsPlot.plot_surface(X, Y, self.weightsData, rstride=1, cstride=1, cmap=cm.coolwarm)
                elif k == 'train':
                    if v.get('new', None):
                        self.trainLines = []
                        for i in range(len(v['data'])):
                            self.trainLines.append(self.trainPlot.plot(v['data'][i][0], v['data'][i][1])[0])
                    else:
                        for i in range(len(v['data'])):
                            self.trainLines[i].set_xdata(np.append(self.trainLines[i].get_xdata(), v['data'][i][0]))
                            self.trainLines[i].set_ydata(np.append(self.trainLines[i].get_ydata(), v['data'][i][1]))

            plt.draw()
            plt.pause(.2)

            self.tasskQueue.task_done()

        plt.ioff()
        plt.show()

        return

    def plot_filters(self, filters, x, y):
        fig = plt.figure()
        for j in range(len(filters)):
            ax = fig.add_subplot(y, x, j)
            ax.matshow(filters[j][0], cmap=cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
        plt.tight_layout()
        return plt