import h5py
import pandas as pd
import numpy as np
from tqdm import trange

class Sigmoid:
    @classmethod
    def f(cls, x):
        return 1/(1 + np.exp(-x))

    @classmethod
    def d(cls, x):
        f = cls.f(x)
        return f * (1 - f)

class Tanh:
    @classmethod
    def f(cls, x):
        return np.tanh(x)

    @classmethod
    def d(cls, x):
        f = cls.f(x)
        return 1 - f**2

class NN:
    """
    Todo:
    - add ability to change activation functions for hidden layers
    - add ability to provide a custom cost function
    - add support for regularisation
    """
    # n is the number of neurons in each layer
    def __init__(self, n, act):
        self.n = n
        self.act = act
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

    def predict(self, xs):
        (z, a) = self.forward(xs)
        return a[-1]

    def loss(self, xs, ys):
        y_hat = self.predict(xs).T
        diff = np.sum(ys*np.log(y_hat) + (1-ys)*np.log(1-y_hat))
        return -diff / len(xs)

    def onehot_acc(self, xs, ys):
        y_hat = self.predict(xs).T
        return np.average(np.argmax(ys, axis=1) == np.argmax(y_hat, axis=1))

    def forward(self, xs):
        # forward pass
        z = [None]
        a = [xs.T]

        for i in range(len(self.weights)):
            z.append(np.dot(self.weights[i], a[i]) + self.bias[i])
            a.append(self.act.f(z[i + 1]))

        return (z, a)

    def backward(self, z, a, ys):
        # backward pass
        m = a[0].shape[1]
        dz = [None for i in range(len(z))]
        dw = [None for i in range(len(z))]
        db = [None for i in range(len(z))]

        dz[-1] = a[-1] - ys.T
        for i in reversed(range(len(self.weights))):
            ii = i + 1
            if dz[ii] is None:
                dz[ii] = np.dot(self.weights[ii].T, dz[ii + 1]) * self.act.d(z[ii])
            dw[ii] = (1/m) * np.dot(dz[ii], a[ii - 1].T)
            db[ii] = (1/m) * np.sum(dz[ii], axis=1, keepdims=True)

        return (dw, db)

            
def batch(xs, ys, group_size):
    for i in range(0, len(xs), group_size):
        yield (xs[i:i+group_size], ys[i:i+group_size])

class GradientDescent:
    def __init__(self, model, xs, ys):
        self.m = model
        self.xs = xs
        self.ys = ys

    def train(self, epochs=1, lr=0.0001, dropout=0):
        epochs = trange(epochs, desc="BasicO")
        for i in epochs:
            epochs.set_description('BasicO (loss=%g)' % self.m.loss(self.xs, self.ys))
            (z, a) = self.m.forward(self.xs)
            if dropout > 0:
                keep = 1 - dropout
                a = [np.multiply(np.random.randn(ac.shape[0], ac.shape[1]) < keep, ac) / keep for ac in a]
            (dw, db) = self.m.backward(z, a, self.ys)
            for i in range(len(self.m.weights)):
                self.m.weights[i] -= (lr * dw[i + 1])
            for i in range(len(self.m.bias)):
                self.m.bias[i] -= (lr * db[i + 1])

class Adam:
    def __init__(self, model, xs, ys, batch_size = 0):
        self.m = model
        if batch_size == 0:
            self.batches = [(xs, ys)]
        else:
            self.batches = [(xs[i:i + batch_size], ys[i:i + batch_size]) for i in range(0, len(xs), batch_size)]
        self.dw = None
        self.db = None

    def calculate_momentum(self, dw, db, beta):
        if self.dw is None:
            self.dw = dw
        if self.db is None:
            self.db = db
        for i in range(len(self.dw))[1:]:
            self.dw[i] = (self.dw[i] * beta) + (dw[i] * (1 - beta))
        for i in range(len(self.db))[1:]:
            self.db[i] = (self.db[i] * beta) + (db[i] * (1 - beta))


    def train(self, epochs=1, lr=0.0001, dropout=0, momentum=0.9):
        epochs = trange(epochs, desc="BasicO")
        for i in epochs:
            for (xs, ys) in self.batches:
                epochs.set_description('Adam (acc=%g)' % self.m.onehot_acc(xs, ys))
                (z, a) = self.m.forward(xs)
                if dropout > 0:
                    keep = 1 - dropout
                    a = [np.multiply(np.random.randn(ac.shape[0], ac.shape[1]) < keep, ac) / keep for ac in a]
                (dw, db) = self.m.backward(z, a, ys)
                self.calculate_momentum(dw, db, momentum)
                   
                for i in range(len(self.m.weights)):
                    self.m.weights[i] -= (lr * dw[i + 1])
                for i in range(len(self.m.bias)):
                    self.m.bias[i] -= (lr * db[i + 1])


if __name__ == "__main__":
    MNIST_data = h5py.File("./data/mnist.hdf5", 'r')
    x_train = np.float32(MNIST_data['x_train'][:])
    y_train = np.int32(np.array(MNIST_data['y_train'][:, 0])).reshape(-1, 1)
    x_test  = np.float32(MNIST_data['x_test'][:])
    y_test  = np.int32(np.array(MNIST_data['y_test'][:, 0])).reshape(-1, 1)
    MNIST_data.close()

    X = np.vstack((x_train, x_test))
    y = np.vstack((y_train, y_test))

    digits = 10
    examples = y.shape[0]
    y = y.reshape(1, examples)
    Y_new = np.eye(digits)[y.astype('int32')]
    Y_new = Y_new.T.reshape(digits, examples)

    # number of training set
    m = 60000
    m_test = X.shape[0] - m
    X_train, X_test = X[:m], X[m:]
    Y_train, Y_test = Y_new[:, :m].T, Y_new[:, m:].T

    nn = NN([784,1000,1000,10], Tanh)
    opt = Adam(nn, X_train, Y_train, batch_size=2048)
    opt.train(epochs=100, lr=0.001)


