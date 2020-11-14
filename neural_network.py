import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

class NeuralNetwork:
    # s is the number of units in each layer, as a numpy vecotor.
    # s[0] is the number of input units. s[-1] is the number of output 
    # units.
    def __init__(self, s):
        self.s = s
        assert len(s) >= 2
        self.weights = self.init_weights()

    def n_params(self):
        return sum([w.size for w in self.weights])

    # given network shape, initialise network weights.
    def init_weights(self):
        weights = []
        for i in range(len(self.s) - 1):
            weights.append(np.random.rand(self.s[i + 1], self.s[i] + 1))
        return weights

    # given network shape, initialise network weights.
    def zero_weights(self):
        weights = []
        for i in range(len(self.s) - 1):
            weights.append(np.zeros([self.s[i + 1], self.s[i] + 1]))
        return weights

    # Run a forward pass of values x
    # on the network. Returns output
    # values y.
    def forward_pass(self, x):
        zs = [np.asarray(x)]
        values = x
        for theta in self.weights:
            v = np.insert(values, 0, 1)
            z = np.matmul(theta, v)
            values = self.activation(z)
            zs.append(z)
        return zs

    def predict(self, x):
        return self.activation(self.forward_pass(x)[-1])

    def cost(self, xs, ys):
        sum = 0
        for i in range(len(xs)):
            err_sum = 0
            y = ys[i]
            h_y = self.predict(xs[i])
            for x in range(len(h_y)):
                err_sum += y[x] * -np.log(h_y[x]) + -(1 - y[x]) * np.log(1 - h_y[x])

            sum += err_sum

        return sum / len(xs)

    # Run one pass of backpropagation
    def back_propagation(self, xs, ys, lr=0.0001):
        assert len(xs) == len(ys)
        delta = self.zero_weights()
        for i in range(len(xs)):
            zs = self.forward_pass(xs[i])
            error = [np.zeros(x.shape) for x in zs]
            error[-1] = self.activation(zs[-1]) - ys[i]
            for l in reversed(range(len(zs) - 1)):
                error[l] = np.matmul(np.atleast_2d(self.weights[l]).T, error[l + 1])[1:] * self.activation_derivative(zs[l])

            for l in range(len(zs) - 1):
                delta[l] += np.matmul(np.atleast_2d(error[l + 1]).T, np.atleast_2d(np.insert(zs[l], 0, 1)))

        for w in range(len(self.weights)):
           self.weights[w] -= delta[w] * lr * 1/len(xs)

    # Train the network
    def train(self, xs, ys, epochs=1000, lr=0.1, plot=False):
        costs = []
        for _ in tqdm(range(epochs)):
            costs.append(self.cost(xs, ys))
            self.back_propagation(xs, ys, lr=lr)

        if plot:
            plt.plot(range(epochs), costs, 'ro')
            plt.show()

    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        act = self.activation(x)
        return np.multiply(act, (1 - act))



if __name__ == "__main__":
    n = NeuralNetwork([100,100,100,10])
    xs = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    ys = [[0], [1], [0], [1], [0], [1], [0], [1], [0], [1]]
    n.train(xs, ys, epochs=100000)

    print(n.predict(xs[0]), n.predict(xs[1]), n.predict(xs[2]), n.predict(xs[3]))

