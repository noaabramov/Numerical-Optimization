import unittest
import numpy as np
import matplotlib.pyplot as plt
from src.unconstrained_min import minimize
from src.utils import plot_contour, plot_values
#from tests.examples import quadratic_1, quadratic_2, quadratic_3, rosenbrock, linear, exp_function

methods = ['Gradient Descent', 'Newton', 'BFGS', 'SR1']

class TestUnconstrainedMin(unittest.TestCase):
    def test_quadratic_1(self):
        """
        Test case for the Quadratic #1 objective function.
        """
        x0 = np.array([1.0, 1.0])
        for method in methods:
            final_location, final_objective, success, path = minimize(quadratic_1, x0, obj_tol=1e-12, param_tol=1e-8, max_iter=100, method=method)
            values = {method: path['values']}
            paths = {method: path['path']}

        plot_contour(quadratic_1, x_lim=(-2, 2), y_lim=(-2, 2), title='Contour of Quadratic #1 Objective Function', paths=paths)
        plot_values({'Optimization Path': values}, title='Function Values per Iteration of Quadratic #1 Objective Function')


    def test_quadratic_2(self):
        """
        Test case for the Quadratic #2 objective function.
        """
        x0 = np.array([1.0, 1.0])
        for method in methods:
            final_location, final_objective, success, path = minimize(quadratic_2, x0, obj_tol=1e-12, param_tol=1e-8, max_iter=100, method=method)
            values = {method: path['values']}
            paths = {method: path['path']}

        plot_contour(quadratic_1, x_lim=(-2, 2), y_lim=(-2, 2), title='Contour of Quadratic #2 Objective Function', paths=paths)
        plot_values({'Optimization Path': values}, title='Function Values per Iteration of Quadratic #2 Objective Function')

    def test_quadratic_3(self):
        """
        Test case for the Quadratic #3 objective function.
        """
        x0 = np.array([1.0, 1.0])
        for method in methods:
            final_location, final_objective, success, path = minimize(quadratic_3, x0, obj_tol=1e-12, param_tol=1e-8, max_iter=100, method=method)
            values = {method: path['values']}
            paths = {method: path['path']}

        plot_contour(quadratic_1, x_lim=(-2, 2), y_lim=(-2, 2), title='Contour of Quadratic #3 Objective Function', paths=paths)
        plot_values({'Optimization Path': values}, title='Function Values per Iteration of Quadratic #3 Objective Function')

    def test_rosenbrock(self):
        """
        Test case for the Rosenbrock objective function.
        """
        x0 = np.array([-1.0, 2.0])
        for method in methods:
            final_location, final_objective, success, path = minimize(rosenbrock, x0, obj_tol=1e-12, param_tol=1e-8, max_iter=10000, method=method)
            values = {method: path['values']}
            paths = {method: path['path']}

        plot_contour(quadratic_1, x_lim=(-2, 2), y_lim=(-2, 5), title='Contour of Rosenbrock Objective Function', paths=paths)
        plot_values({'Optimization Path': values}, title='Function Values per Iteration of Rosenbrock Objective Function')

    def test_linear(self):
        """
        Test case for the Linear objective function.
        """
        x0 = np.array([1.0, 1.0])
        for method in methods:
            final_location, final_objective, success, path = minimize(linear, x0, obj_tol=1e-12, param_tol=1e-8, max_iter=100, method=method)
            values = {method: path['values']}
            paths = {method: path['path']}

        plot_contour(quadratic_1, x_lim=(-300, 2), y_lim=(-300, 2), title='Contour of Linear Objective Function', paths=paths)
        plot_values({'Optimization Path': values}, title='Function Values per Iteration of Linear Objective Function')

    def test_exp_function(self):
        """
        Test case for the Exponential objective function.
        """
        x0 = np.array([1.0, 1.0])
        for method in methods:
            final_location, final_objective, success, path = minimize(exp_function, x0, obj_tol=1e-12, param_tol=1e-8, max_iter=100, method=method)
            values = {method: path['values']}
            paths = {method: path['path']}

        plot_contour(quadratic_1, x_lim=(-1, 1), y_lim=(-1, 1), title='Contour of Exponential Objective Function', paths=paths)
        plot_values({'Optimization Path': values}, title='Function Values per Iteration of Exponential Objective Function')

if __name__ == '__main__':
    unittest.main()



def quadratic_1(x, compute_hessian):
    Q = np.array([[1, 0], [0, 1]])
    f = x.T.dot(Q).dot(x)
    g = 2 * Q.dot(x)
    h = 2 * Q if compute_hessian else None
    return f, g, h


def quadratic_2(x, compute_hessian):
    Q = np.array([[1, 0], [0, 100]])
    f = x.T.dot(Q).dot(x)
    g = 2 * Q.dot(x)
    h = 2 * Q if compute_hessian else None
    return f, g, h


def quadratic_3(x, compute_hessian):
    Q1 = np.array([[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]])
    Q2 = np.array([[100, 0], [0, 1]])
    Q_final = Q1.T.dot(Q2).dot(Q1)

    f = x.T.dot(Q_final).dot(x)
    g = 2 * Q_final.dot(x)
    h = 2 * Q_final if compute_hessian else None
    return f, g, h


def rosenbrock(x, compute_hessian):
    f = 100 * ((x[1] - (x[0] ** 2)) ** 2) + ((1 - x[0]) ** 2)
    g = np.array([-400 * x[0] * x[1] + 400 * (x[0] ** 3) + 2 * x[0] - 2,
                  200 * x[1] - 200 * (x[0] ** 2)])
    h = None
    if compute_hessian:
        h = np.array([[-400 * x[1] + 1200 * (x[0] ** 2) + 2, -400 * x[0]],
                      [-400 * x[0],         200]])
    return f, g, h


def linear(x, compute_hessian):
    a = np.array([3, 3])
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

