__author__ = 'shiriiin
'

from SimplePerceptron import SimplePerceptron, shuffle, np, plotter


class SecondOrderPerceptron(SimplePerceptron):
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

            tmp_l = []
            for d in data[0]:
                tmp_l.append(d**2)
            data[0].extend(tmp_l)

        print data_set
        shuffle(data_set)
        self.input_training = data_set[:(int(len(data_set) * training_percentage))]
        self.input_validation = data_set[(int(len(data_set) * training_percentage)):]

    def update_plot(self, input_dimensions, output_dimensions, validation_error, training_error):
        if output_dimensions == 1 and input_dimensions == 4:
            def line_formula(x, o):
                res = []
                m1 = self.weight[o][2]**2/(4*self.weight[o][4]**2)
                m0 = -self.weight[o][2]/(2*self.weight[o][4])

                for xi in x:
                    ri = np.sqrt(((xi**2)*self.weight[o][3]+xi*self.weight[o][1]+self.weight[o][0])/-self.weight[o][4]+m1)
                    res.append(ri)
                tmp_l = [-r for r in res]
                res.extend(tmp_l)
                res = [r+m0 for r in res]
                return res
            tmp_range = np.linspace(-50, 50, 100)
            t2 = tmp_range.tolist()
            t2r = tmp_range.tolist()
            t2r.reverse()
            t2.extend(t2r)

            self.plot.update_sample_plotter(x=t2, y=line_formula(tmp_range, 0))
        self.plot.update_error_plotter(validation_error, training_error)
        self.plot.update_weights_plotter([w for weights in self.weight for w in weights])