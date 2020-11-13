import numpy as np


class NeuralNetwork:
    # s is the number of units in each layer, as a numpy vecotor.
    # s[0] is the number of input units. s[-1] is the number of output 
    # units.
    def __init__(self, s):
        self.s = s
        assert len(s) >= 2
        self.weights = self.init_weights()

    # given network shape, initialise network weights.
    def init_weights(self):
        weights = []
        for i in range(len(self.s) - 1):
            weights.append(np.ones([self.s[i + 1], self.s[i] + 1]))
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

    def back_propagation(self, xs, ys, lr=0.001):
        assert len(xs) == len(ys)
        delta = self.init_weights()
        for i in range(len(xs)):
            zs = self.forward_pass(xs[i])
            # initialise error tensor
            # to all zeros, and same shape as
            # network values.
            error = [np.zeros(x.shape) for x in zs]
            error[-1] = self.activation(zs[-1]) - ys[i]
            for l in reversed(range(len(zs) - 1)[1:]):
                lhs = np.matmul(np.atleast_2d(self.weights[l]).T, error[l + 1])[1:]
                rhs = self.activation_derivative(zs[l])
                error[l] = lhs * rhs

            # There is some sort of bug here...
            for l in range(len(zs) - 1):
                delta[l] += np.matmul(np.atleast_2d(error[l + 1]).T, np.atleast_2d(np.insert(zs[l], 0, 1)))

        for w in range(len(self.weights)):
           self.weights[w] -= delta[w] * lr * 1/len(xs)

    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        return self.activation(x) * (1 - self.activation(x))

if __name__ == "__main__":
    n = NeuralNetwork([2,5,5,1])
    xs = [[0,0], [0,1], [1,0], [1,1]]
    ys = [[0], [1], [1], [0]]

    for i in range(10000):
        print(n.predict(xs[0]) + n.predict(xs[3]) - n.predict(xs[1]) - n.predict(xs[2]))
        n.back_propagation(xs, ys)

    print(n.predict(xs[0]))
    print(n.predict(xs[1]))
    print(n.predict(xs[2]))
    print(n.predict(xs[3]))
