import numpy as np


def quadratic_1(x, compute_hessian):
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    f = x.T.dot(Q).dot(x)
    g = 2 * Q.dot(x)
    h = 2 * Q if compute_hessian else None
    return f, g, h


def quadratic_2(x, compute_hessian):
    Q = np.array([[1.0, 0.0], [0.0, 100.0]])
    f = x.T.dot(Q).dot(x)
    g = 2 * Q.dot(x)
    h = 2 * Q if compute_hessian else None
    return f, g, h


def quadratic_3(x, compute_hessian):
    Q1 = np.array([[float(np.sqrt(3) / 2), -0.5], [0.5, float(np.sqrt(3) / 2)]])
    Q2 = np.array([[100.0, 0.0], [0.0, 1.0]])
    Q_final = Q1.T.dot(Q2).dot(Q1)

    f = x.T.dot(Q_final).dot(x)
    g = 2 * Q_final.dot(x)
    h = 2 * Q_final if compute_hessian else None
    return f, g, h


def rosenbrock(x, compute_hessian):
    f = 100 * ((x[1] - (x[0] ** 2)) ** 2) + ((1 - x[0]) ** 2)
    g = np.array([float(-400 * x[0] * x[1] + 400 * (x[0] ** 3) + 2 * x[0] - 2),
                  float(200 * x[1] - 200 * (x[0] ** 2))])
    h = None
    if compute_hessian:
        h = np.array([[float(-400 * x[1] + 1200 * (x[0] ** 2) + 2, -400 * x[0])],
                      [float(-400 * x[0]),         200.0]])
    return f, g, h


def linear(x, compute_hessian):
    a = np.array([3.0, 3.0])
    f = a.T.dot(x)
    g = a
    h = None
    if compute_hessian:
        h = np.zeros((2, 2))
    return f, g, h


def exp_function(x, compute_hessian):
    f = np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] -
                                               3 * x[1] - 0.1) + np.exp(-x[0] - 0.1)
    g = np.array([np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) - np.exp(-x[0] - 0.1),
                  3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)])
    h = None
    if compute_hessian:
        h = np.array([[np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) + np.exp(-x[0] - 0.1),
                       3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)],
                      [3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1),
                       9 * np.exp(x[0] + 3 * x[1] - 0.1) + 9 * np.exp(x[0] - 3 * x[1] - 0.1)]])
    return f, g, h
