__author__ = 'shiriiin'

from SimplePerceptron import SimplePerceptron, shuffle, np, plotter


class Adaline(SimplePerceptron):
    def train(self, train_set):
        self.is_finished = False
        counter = 0

        if getattr(self, 'weight', None) is None:
            self.weight = np.random.randn(len(train_set[0][1]), 1 + len(train_set[0][0]))

        # plot section
        if len(train_set[0][1]) == 1:
            self.plot.refresh_sample_plotter([self.input_training, self.input_validation])
        self.plot.refresh_error_plotter(validation_error=self.test(train_set[:2]))
        self.plot.refresh_weights_plotter([w for weights in self.weight for w in weights])
        #

        for ep in range(self.epochs):
            errors = [0] * (len(self.weight)+1)
            for Xi, Yi in train_set:
                tmp_y = [0] * len(Yi)
                for j in range(len(self.weight)):
                    tmp_error = (Yi[j] - self.net_input(Xi, j))
                    update = self.eta * tmp_error
                    # print tmp_error , self.net_input(Xi, j)
                    # print update, update * np.array(Xi), self.predict(Xi, j)
                    self.weight[j][1:] += update * np.array(Xi)
                    self.weight[j][0] += update
                    tmp_y = self.net_input(Xi, j)
                ind = Yi.index(1)+1 if 1 in Yi else 0
                errors[ind] += (np.subtract(Yi, tmp_y)**2/2).sum()
            self.training_errors.append(errors)

            validation_error = self.test(self.input_validation)
            print "validation error: %s" % str(validation_error), "\t training error: %s" % str(errors)

            self.update_plot(len(train_set[0][0]), len(train_set[0][1]), validation_error, errors)

            if validation_error <= self.best_validation_errors:
                self.best_validation_errors = validation_error
                self.best_weight = self.weight
                counter = 0
            else:
                counter += 1

            # if counter >= 100:
            #     self.weight = self.best_weight
            #     print "break the loop because of over fitting at %d" % ep
            #     break

            if validation_error <= self.error_threshold:
                print "break the loop because of error satisfying at %d" % ep
                break

            if sum(errors) == 0:
                print "break the loop because of training error 0 at %d" % ep
                break

        print "weights", self.weight
        self.is_finished = True

    def test(self, test_set):
        errors = [0] * (len(self.weight)+1)
        for Xi, Yi in test_set:
            tmp_y = [0] * len(Yi)
            for j in range(len(self.weight)):
                tmp_y[j] = self.net_input(Xi, j)
            ind = Yi.index(1)+1 if 1 in Yi else 0
            errors[ind] += (np.subtract(Yi, tmp_y)**2/2).sum()
        self.validation_errors.append(errors)
        return errors




# if __name__ == '__main__':
    # simple = SimplePerceptron(epochs=1000, eta=0.0001)
    # simple.get_data('Dataset1.data')
    # # simple.train([([2,2],[-1,1])])
    # simple.train(simple.training_set)
    # print simple.errors
    # print simple.weight

