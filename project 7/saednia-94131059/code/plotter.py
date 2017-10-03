__author__ = 'shiriiin'

import multiprocessing
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


class Plotter(multiprocessing.Process):
    def __init__(self, tasks_queue):
        multiprocessing.Process.__init__(self)
        self.daemon = True
        self.task_queue = tasks_queue

        self.figure = plt.figure()

        self.weights_plot = plt.subplot2grid((6, 3), (0, 0), rowspan=3, colspan=3, projection='3d')
        # self.weights_plot = self.figure.add_subplot(3, 1, 1, projection='3d')
        # self.weights_plot.axis([0, 60, -10, 10])
        # self.weights_plot.set_zlim(-40, 40)
        self.weight_line = None
        self.weights_data = None

        self.series_plot = plt.subplot2grid((6, 1), (3, 0), rowspan=1, colspan=1)
        # self.series_plot = self.figure.add_subplot(3, 1, 2)
        self.series_plot.axis([0, 10000, -.3, .3])
        self.series_line = None

        self.error_plot = plt.subplot2grid((6, 1), (4, 0), rowspan=1, colspan=1)
        # self.error_plot = self.figure.add_subplot(3, 1, 3)
        self.error_plot.axis([0, 200, 0, 100])
        self.error_line = None

        plt.grid()
        plt.ion()
        plt.show()

    def run(self):
        ep = 1

        while 1:
            try:
                task = self.task_queue.get(timeout=.01)
            except Exception:
                plt.draw()
                plt.pause(.01)
                continue

            if task is None:
                self.task_queue.task_done()
                break

            for k, v in task.iteritems():
                if k == 'weights':
                    if v.get('new', None):
                        if self.weight_line is not None:
                            self.weight_line.remove()
                        self.weights_data = []
                        self.weights_data.append([v['data'][i][1] for i in range(len(v['data']))])
                        x, y = np.meshgrid(range(len(v['data'])), range(v['data'][0][0]+1))
                        self.weight_line = self.weights_plot.plot_surface(x, y, self.weights_data, rstride=1, cstride=1, cmap=cm.coolwarm)
                    else:
                        self.weight_line.remove()
                        self.weights_data.append([v['data'][i][1] for i in range(len(v['data']))])
                        x, y = np.meshgrid(range(len(v['data'])), range(v['data'][0][0]+1))
                        # print range(len(v['data'])), range(v['data'][0][0]+1)
                        # print X, Y, self.weightsData
                        self.weight_line = self.weights_plot.plot_surface(x, y, self.weights_data, rstride=1, cstride=1, cmap=cm.coolwarm)
                elif k == 'error':
                    if self.error_line is None:
                        self.error_line = self.error_plot.plot(ep, v['data'])[0]
                    else:
                        self.error_line.set_xdata(np.append(self.error_line.get_xdata(), ep))
                        self.error_line.set_ydata(np.append(self.error_line.get_ydata(), v['data']))
                elif k == 'series':
                    if ep == 1:
                        self.series_plot.plot(range(1, len(v['main'])+1), v['main'])
                        self.series_line = self.series_plot.plot(range(1, len(v['data'])+1), v['data'])[0]
                    else:
                        self.series_line.remove()
                        self.series_line = self.series_plot.plot(range(1, len(v['data'])+1), v['data'], 'g')[0]

            plt.draw()
            plt.pause(.2)

            self.task_queue.task_done()

            ep += 1

        plt.ioff()
        plt.show()

        return