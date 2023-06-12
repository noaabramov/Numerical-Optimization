import matplotlib.pyplot as plt
import numpy as np


def plot_contour(f, x_lim, y_lim, title, paths={}, levels=100):
    """
    Plot a contour plot of a 2D function.

    Parameters:
    - f: The function to plot.
    - x_lim: The limits for the x-axis.
    - y_lim: The limits for the y-axis.
    - title: The title of the plot.
    - paths: Optional dictionary of paths to be plotted.
    - levels: The number of contour levels to display.

    Returns:
    None
    """
    x = np.linspace(x_lim[0], x_lim[1])
    y = np.linspace(y_lim[0], y_lim[1])

    xs, ys = np.meshgrid(x, y)
    f_vals = np.vectorize(lambda x1, x2: f(
        np.array([x1, x2]), False)[0])(xs, ys)

    fig, ax = plt.subplots(1, 1)
    contour = ax.contourf(x, y, f_vals, levels)
    fig.colorbar(contour)
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    
    if len(paths) > 0:
        for name, path in paths.items():
            plt.plot(path[:, 0], path[:, 1], label=name)
        plt.legend()
    
    plt.show()


def plot_values(values_dict, title):
    """
    Plot the values of different variables over iterations.

    Parameters:
    - values_dict: A dictionary containing the values of variables.
    - title: The title of the plot.

    Returns:
    None
    """
    fig, ax = plt.subplots(1, 1)
    for name, values in values_dict.items():
        x = np.linspace(0, len(values)-1, len(values))
        ax.plot(x, values, marker='.', label=name)
    ax.set_title(title)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Function values')
    plt.legend()
    plt.show()
