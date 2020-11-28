import pandas as pd
import numpy as np

class SigmoidActivation:
    @classmethod
    def f(cls, x):
        return 1/(1 + np.exp(-x))

    @classmethod
    def d(cls, x):
        f = cls.f(x)
        return f * (1 - f)


class NN:
    """
    Todo:
    - add ability to change activation functions for hidden layers
    - add ability to provide a custom cost function
    - add support for regularisation
    - add support for mini-batching
    """
    # n is the number of neurons in each layer
    def __init__(self, n):
        self.n = n
        assert len(n) >= 2
        self.weights = self.init_weights()
        self.bias = self.init_bias()

    def init_weights(self):
        weights = []
        for i in range(len(self.n) - 1):
            weights.append(np.random.randn(self.n[i + 1], self.n[i]) * 0.01)
        return weights

    def init_bias(self):
        bias = []
        for i in range(len(self.n) - 1):
            bias.append(np.zeros((self.n[i + 1], 1)))
        return bias
        
    def n_params(self):
        return sum([w.size for w in self.weights] + [b.size for b in self.bias])

    def forward(self, xs):
        # forward pass
        z = [None]
        a = [xs.T]

        for i in range(len(self.weights)):
            z.append(np.dot(self.weights[i], a[i]) + self.bias[i])
            a.append(SigmoidActivation.f(z[i + 1]))

        return (z, a)

    def predict(self, xs):
        (z, a) = self.forward(xs)
        return a[-1]

    def loss(self, xs, ys):
        y_hat = self.predict(xs)
        diff = np.sum(y_hat - ys.T)
        return (diff * diff) / len(xs)

    def backprop(self, xs, ys, alpha=0.01, dropout=0):
        weights = [np.copy(w) for w in self.weights]
        bias = [np.copy(b) for b in self.bias]

        (z, a) = self.forward(xs)

        if dropout > 0:
            keep = 1 - dropout
            a = [np.multiply(np.random.randn(ac.shape[0], ac.shape[1]) < keep, ac) / keep for ac in a]

        # backward pass
        m = xs.shape[0]
        dz = [None for i in range(len(z))]
        dw = [None for i in range(len(z))]
        db = [None for i in range(len(z))]

        dz[-1] = a[-1] - ys.T
        for i in reversed(range(len(weights))):
            ii = i + 1
            if dz[ii] is None:
                dz[ii] = np.dot(weights[ii].T, dz[ii + 1]) * SigmoidActivation.d(z[ii])
            dw[ii] = (1/m) * np.dot(dz[ii], a[ii - 1].T)
            db[ii] = (1/m) * np.sum(dz[ii], axis=1, keepdims=True)


        for i in range(len(weights)):
            weights[i] -= (alpha * dw[i + 1])

        self.weights = weights
        self.bias = bias
            


if __name__ == "__main__":
    titanic = pd.read_csv("data/train.csv")[["Age", "Sex", "Survived"]].dropna()
    sex = [0 if x == "male" else 1 for x in titanic["Sex"]]
    age = titanic["Age"]
    xs = np.asarray([age, sex]).T
    ys = np.asarray([[x] for x in titanic["Survived"]])
    nn = NN([2,30,30,30,30,1])

    for i in range(10000):
        nn.backprop(xs, ys, dropout=0.1)
        print(nn.loss(xs, ys))


