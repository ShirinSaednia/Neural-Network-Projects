__author__ = 'shiriiin'

import csv
import numpy as np


def readData(regul, const):
    training_data = []
    validation_data = []

    max_l = [-1000]*384
    min_l = [+1000]*384

    csvfile = open('slice_localization_data.csv', 'rb')
    spamreader = csv.reader(csvfile, delimiter=',')
    i = 0
    for line in spamreader:
        i += 1

        if i == 1:
            continue

        row = [[np.float32(d) for d in line[1:-1]], np.reshape(np.float32(line[-1]), (1, 1))]
        row[0] = np.array(row[0], np.float32).reshape(384, 1)
        if i <= 37450:
            training_data.append(row)
        else:
            validation_data.append(row)

        if regul:
            for j in xrange(384):
                if min_l[j] > row[0][j][0]:
                    min_l[j] = row[0][j][0]
                if max_l[j] < row[0][j][0]:
                    max_l[j] = row[0][j][0]

    if regul:
        delta_l = [l1-l2 for l1, l2 in zip(max_l, min_l)]
        for i in xrange(len(delta_l)):
            if delta_l[i] == 0: delta_l[i] = 1

        for r in xrange(len(training_data)):
            for i in xrange(384):
                training_data[r][0][i] = const*(training_data[r][0][i]-min_l[i])/delta_l[i]
        for r in xrange(len(validation_data)):
            for i in xrange(384):
                validation_data[r][0][i] = const*(validation_data[r][0][i]-min_l[i])/delta_l[i]

    csvfile.close()

    return training_data, validation_data