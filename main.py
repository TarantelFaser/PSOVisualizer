import numpy as np
import matplotlib.pyplot as plt

AXISMIN = -10
AXISMAX = 10
DOTS = 100
ARROWSIZE = 25


# takes arrays of coordinates of particles and the velocities
# and a function and renders the plot
def renderPlot(partXs, partYs, partXvel, partYvel, func):
    fig, ax = plt.subplots()
    plt.xlabel('X')
    plt.ylabel('Y')

    # get the contour plot for the function (func : R^2 -> R)
    xf = np.linspace(AXISMIN, AXISMAX, DOTS)
    yf = np.linspace(AXISMIN, AXISMAX, DOTS)
    Xf, Yf = np.meshgrid(xf, yf)
    cf = ax.contourf(Xf, Yf, func(Xf, Yf))
    fig.colorbar(cf, ax=ax)

    # draw the velocity arrows
    ax.quiver(partXs, partYs, partXvel, partYvel, scale=ARROWSIZE)
    ax.set_title('Particle Swarm Optimization')
    ax.set_aspect('equal')
    ax.axis([AXISMIN, AXISMAX, AXISMIN, AXISMAX])

    plt.show()

    # TODO marker (Kreuze oder so im Diagramm)
    # TODO animation oder alternativ viele einzelne PNGs zusammen zu einem GIF

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



if __name__ == '__main__':
    # positions of particles
    X = [1, 2, 3, 4]
    Y = [5, 2, 8, 1]

    # velocities of particles
    xVel = [1, 2, 3, -2]
    yVel = [2, 5, -3, 0]

    renderPlot(X, Y, xVel, yVel, quadratic_function)