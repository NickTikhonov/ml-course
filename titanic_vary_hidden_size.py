import numpy as np
import pandas as pd
import neural_network
from matplotlib import pyplot as plt

if __name__ == "__main__":
    titanic = pd.read_csv("data/train.csv")[["Age", "Sex", "Survived"]].dropna()
    sex = [0 if x == "male" else 1 for x in titanic["Sex"]]
    age = titanic["Age"]
    xs = np.asarray([age, sex]).T
    ys = np.asarray([[x] for x in titanic["Survived"]])

    for l in range(1, 20, 3):
        n = neural_network.NeuralNetwork([2,l,1])
        ep = []
        costs = []
        def cb(epoch, cost):
            ep.append(epoch)
            costs.append(cost)

        n.train(xs, ys, epochs=400, lr=0.0005, cb=cb)
        plt.plot(ep, costs)

    plt.show()

