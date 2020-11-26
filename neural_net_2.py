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
        z1 = np.dot(self.weights[0], xs.T) + self.bias[0]
        a1 = SigmoidActivation.f(z1)
        z2 = np.dot(self.weights[1], a1) + self.bias[1]
        a2 = SigmoidActivation.f(z2)
        return a2

    def loss(self, xs, ys):
        y_hat = self.forward(xs)
        diff = np.sum(y_hat - ys.T)
        return (diff * diff) / len(xs)

    def backprop_1l(self, xs, ys, alpha=0.01):
        # forward pass
        z1 = np.dot(self.weights[0], xs.T) + self.bias[0]
        a1 = SigmoidActivation.f(z1)
        z2 = np.dot(self.weights[1], a1) + self.bias[1]
        a2 = SigmoidActivation.f(z2)

        # backward pass
        m = xs.shape[0]
        dz2 = a2 - ys.T
        dw2 = (1/m) * np.dot(dz2, a1.T)
        db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.dot(self.weights[1].T, dz2) * SigmoidActivation.d(z1)
        dw1 = (1/m) * np.dot(dz1, xs)
        db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)

        # update weights
        self.weights[1] = self.weights[1] - (alpha * dw2)
        self.bias[1] = self.bias[1] - (alpha * db2)
        self.weights[0] = self.weights[0] - (alpha * dw1)
        self.bias[0] = self.bias[0] - (alpha * db1)



if __name__ == "__main__":
    titanic = pd.read_csv("data/train.csv")[["Age", "Sex", "Survived"]].dropna()
    sex = [0 if x == "male" else 1 for x in titanic["Sex"]]
    age = titanic["Age"]
    xs = np.asarray([age, sex]).T
    ys = np.asarray([[x] for x in titanic["Survived"]])

    nn = NN([2,10,1])
    for i in range(100):
        print(nn.loss(xs, ys))
        nn.backprop_1l(xs, ys)


