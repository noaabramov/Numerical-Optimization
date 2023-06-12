import unittest
import numpy as np
import matplotlib.pyplot as plt
from src.unconstrained_min import minimize
from src.utils import plot_contour, plot_function_values
#from tests.examples import quadratic_1, quadratic_2, quadratic_3, rosenbrock, linear, exp_function

methods = ['gradient descent', 'newton', 'bfgs', 'sr1']

class TestUnconstrainedMin(unittest.TestCase):
    def test_quadratic_1(self):
        x0 = np.array([1, 1])
        for method in methods:
            final_location, final_objective, success, path = minimize(quadratic_1, x0, obj_tol=1e-12, param_tol=1e-8, max_iter=100, method = method)
            self.assertTrue(success)
            self.assertAlmostEqual(final_location[0], 0.0, places=5)
            self.assertAlmostEqual(final_location[1], 0.0, places=5)

            # Plot the contour with the paths
            plot_contour(f, x_lim=(-5, 5), y_lim=(-5, 5), title='Contour Plot', paths={'Optimization Path': np.array(path)})

            # Plot the function values at each iteration
            function_values = [f(x, False)[0] for x in path]
            plot_function_values({'Optimization Path': function_values}, title='Function Values')

            plot_function_values(final_objective, title='Quadratic 1')

    def test_quadratic_2(self):
        x0 = np.array([1, 1])
        for method in methods:
            final_location, final_objective, success, path = minimize(quadratic_2, x0, obj_tol=1e-12, param_tol=1e-8, max_iter=100, method = method)
            self.assertTrue(success)
            self.assertAlmostEqual(final_location[0], 0.0, places=5)
            self.assertAlmostEqual(final_location[1], 0.0, places=5)

            # Plot the contour with the paths
            plot_contour(f, x_lim=(-5, 5), y_lim=(-5, 5), title='Contour Plot', paths={'Optimization Path': np.array(path)})

            # Plot the function values at each iteration
            function_values = [f(x, False)[0] for x in path]
            plot_function_values({'Optimization Path': function_values}, title='Function Values')

            plot_function_values(final_objective, title='Quadratic 2')
    def test_quadratic_3(self):
        x0 = np.array([1, 1])
        for method in methods:
            final_location, final_objective, success, path = minimize(quadratic_3, x0, obj_tol=1e-12, param_tol=1e-8, max_iter=100, method = method)
            self.assertTrue(success)
            self.assertAlmostEqual(final_location[0], 0.0, places=5)
            self.assertAlmostEqual(final_location[1], 0.0, places=5)

            # Plot the contour with the paths
            plot_contour(f, x_lim=(-5, 5), y_lim=(-5, 5), title='Contour Plot', paths={'Optimization Path': np.array(path)})

            # Plot the function values at each iteration
            function_values = [f(x, False)[0] for x in path]
            plot_function_values({'Optimization Path': function_values}, title='Function Values')

            plot_function_values(final_objective, title='Quadratic 3')

    def test_rosenbrock(self):
        x0 = np.array([-1, 2])
        for method in methods:
            final_location, final_objective, success, path = minimize(rosenbrock, x0, obj_tol=1e-12, param_tol=1e-8, max_iter=10000, method = method)
            self.assertTrue(success)
            self.assertAlmostEqual(final_location[0], 1.0, places=5)
            self.assertAlmostEqual(final_location[1], 1.0, places=5)

            # Plot the function values at each iteration
            function_values = [f(x, False)[0] for x in path]
            plot_function_values({'Optimization Path': function_values}, title='Function Values')

            # Plot function values during optimization
            plot_function_values(final_objective, title='Rosenbrock')

    def test_linear(self):
        x0 = np.array([1, 1])
        for method in methods:
            final_location, final_objective, success, path = minimize(linear, x0, obj_tol=1e-12, param_tol=1e-8, max_iter=100, method = method)
            self.assertTrue(success)
            self.assertAlmostEqual(final_location[0], 0.0, places=5)
            self.assertAlmostEqual(final_location[1], 0.0, places=5)

            # Plot the function values at each iteration
            function_values = [f(x, False)[0] for x in path]
            plot_function_values({'Optimization Path': function_values}, title='Function Values')

            # Plot function values during optimization
            plot_function_values(final_objective, title='Linear')

    def test_exp_function(self):
        x0 = np.array([1, 1])
        for method in methods:    
            final_location, final_objective, success, path = minimize(exp_function, x0, obj_tol=1e-12, param_tol=1e-8, max_iter=100, method = method)
            self.assertTrue(success)
            self.assertAlmostEqual(final_location[0], -0.333333, places=5)
            self.assertAlmostEqual(final_location[1], 0.0, places=5)

            # Plot function values during optimization
            plot_function_values(final_objective, title='Exponential Function')

    if __name__ == '__main__':
        unittest.main()

def quadratic_1(x, compute_hessian):
    Q = np.array([[1, 0], [0, 1]])
    f = np.transpose(x).dot(Q).dot(x)
    g = 2 * Q.dot(x)
    h = 2 * Q if compute_hessian else None
    return f, g, h


def quadratic_2(x, compute_hessian):
    Q = np.array([[1, 0], [0, 100]])
    f = np.transpose(x).dot(Q).dot(x)
    g = 2 * Q.dot(x)
    h = 2 * Q if compute_hessian else None
    return f, g, h


def quadratic_3(x, compute_hessian):
    Q1 = np.array([[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]])
    Q2 = np.array([[100, 0], [0, 1]])
    Q_final = np.transpose(Q1).dot(Q2).dot(Q1)

    f = np.transpose(x).dot(Q_final).dot(x)
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
    f = np.transpose(a).dot(x)
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
