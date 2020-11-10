import numpy as np
import matplotlib.pyplot as plt

def cost(theta, xs, ys):
    err_sum = 0
    hyp = h(theta)
    for i in range(len(xs)):
        err_sum += (hyp(xs[i]) - ys[i])**2
    return err_sum / (2 * len(xs))

def gradient_descent(theta, xs, ys, lr = 0.001):
    assert len(xs) == len(ys)
    new_theta = []
    hyp = h(theta)
    for j in range(len(theta)):
        err_sum = 0
        for i in range(len(xs)):
            err_sum += (hyp(xs[i]) - ys[i])*xs[i][j]
        new_theta.append(theta[j] - (lr * 1/len(xs) * err_sum))
    
    assert len(new_theta) == len(theta)
    return np.asarray(new_theta)


# Linear regression hypothesis 
# function generator.
def h(theta):
    def hyp(x):
        return np.dot(theta, x)

    return hyp

def fit_poly():
    x1 = np.asarray([1,2,3,5,2,3,6,3,3.6,3.4,2.5])
    x2 = x1 * x1
    xs = np.asarray([x1/x1, x1, x2]).T

    ys = x1 * (x1 * 0.2) * 1.5 - 20
    theta = np.asarray([0,0,0])

    for i in range(200000):
        theta = gradient_descent(theta, xs, ys, lr=0.001)
        print(theta)

    hyp = h(theta)
    h_y = list(map(hyp, xs))

    plt.plot(x1, ys, 'ro')
    plt.plot(x1, h_y, 'bo')
    plt.show()

    print(h_y)

if __name__ == "__main__":
    fit_poly()