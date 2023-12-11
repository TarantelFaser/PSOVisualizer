import numpy as np
import matplotlib.pyplot as plt

AXISMIN = -10
AXISMAX = 10
ARROWSIZE = 25


# takes arrays of coordinates of particles and the velocities
# and a function and renders the plot
def renderPlot(partXs, partYs, partXvel, partYvel, func):
    fig, ax = plt.subplots()

    ax.quiver(partXs, partYs, partXvel, partYvel, scale=ARROWSIZE)
    ax.set_title('Quiver plot with one arrow')
    ax.set_aspect('equal')
    ax.axis([AXISMIN, AXISMAX, AXISMIN, AXISMAX])

    # get the contour plot for the function (func : R^2 -> R)
    xf = np.arange(AXISMIN, AXISMAX, 0.1)
    yf = np.arange(AXISMIN, AXISMAX, 0.1)
    Xf, Yf = np.meshgrid(xf, yf)
    ax.contour(Xf, Yf, func(Xf, Yf))

    plt.show()


def rosenbrock_function(x, y):
    a = 1
    b = 100
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def rastrigin_function(x, y):
    A = 10
    return A * 2 + x**2 - A * np.cos(2 * np.pi * x) + y**2 - A * np.cos(2 * np.pi * y)


def quadratic_function(x, y):
    a = 1
    b = 2
    c = 0.5
    d = -1
    e = 1
    f = 0

    return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    X = [1, 2, 3, 4]
    Y = [5, 2, 8, 1]

    xVel = [1, 2, 3, -2]
    yVel = [2, 5, -3, 0]
    renderPlot(X, Y, xVel, yVel, quadratic_function)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
