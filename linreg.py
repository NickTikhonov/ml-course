import numpy as np
import matplotlib.pyplot as plt

# Cost function for linear regression.
def linear_cost(theta, xs, ys):
    err_sum = 0
    hyp = h(theta)
    for i in range(len(xs)):
        err_sum += (hyp(xs[i]) - ys[i])**2
    return err_sum / (2 * len(xs))


# Cost function for logistic regression
def logistic_cost(theta, xs, ys):
    err_sum = 0
    hyp = h_logistic(theta)
    for i in range(len(xs)):
        err_sum += ys[i] * -np.log(hyp(xs[i])) + -(1 - ys[i]) * np.log(1 - hyp(xs[i]))

    return err_sum / (2 * len(xs))


# Gradient descent implementation,
# which minimises MSE of hyp, by adjusting parameters theta
# (this runs a single iteration)
def gradient_descent(theta, hyp, xs, ys, lr = 0.001):
    assert len(xs) == len(ys)
    new_theta = []
    for j in range(len(theta)):
        err_sum = 0
        for i in range(len(xs)):
            err_sum += (hyp(xs[i]) - ys[i])*xs[i][j]
        new_theta.append(theta[j] - (lr * 1/len(xs) * err_sum))
    
    assert len(new_theta) == len(theta)
    return np.asarray(new_theta)


# Linear regression hypothesis 
# function generator.
# Returns a function that predicts a y value.
def h(theta):
    def hyp(x):
        return np.dot(theta, x)

    return hyp


# Logistic regression hypothesis
# function generator.
# Returns a function that predicts a y value between 0 and 1
def h_logistic(theta):
    def hyp(x):
        return 1/(1 + np.exp(-np.dot(theta, x))) 

    return hyp


# Fit a logistic regression example.
def fit_logistic():
    xs = np.asarray(list(map(lambda x: [1, x], np.asarray([1,2,3,4,5]))))
    ys = np.asarray([0,0,0,1,1])
    theta = np.asarray([0.1, 0.1])

    for i in range(10000):
        hyp = h_logistic(theta)
        theta = gradient_descent(theta, hyp, xs, ys, lr=0.001)
        print(logistic_cost(theta, xs, ys), theta)

    hyp = h(theta)
    h_y = list(map(hyp, xs))

    plt.plot(xs, ys, 'ro')
    plt.plot(xs, h_y, 'bo')
    plt.show()


# Fit a linear regression example.
def fit_linear():
    x1 = np.asarray([1,2,3,5,2,3,6,3,3.6,3.4,2.5])
    x2 = x1 * x1
    xs = np.asarray([x1/x1, x1, x2]).T

    ys = x1 * (x1 * 0.2) * 1.5 - 20
    theta = np.asarray([0,0,0])

    for i in range(200000):
        hyp = h(theta)
        theta = gradient_descent(theta, hyp, xs, ys, lr=0.001)
        print(theta)

    hyp = h(theta)
    h_y = list(map(hyp, xs))

    plt.plot(x1, ys, 'ro')
    plt.plot(x1, h_y, 'bo')
    plt.show()

    print(h_y)


if __name__ == "__main__":
    fit_logistic()
