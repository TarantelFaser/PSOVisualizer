import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

AXISMIN = -10
AXISMAX = 10
DOTS = 100
ARROWSIZE = 25
FPS = 25


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



    # arrange returns array of evenly spaced values (this could be replaced by arrays of positions from the pso algorithm)
    x1 = np.arange(0, -2, -0.02)
    y1 = x1 ** 2
    #x1 = np.arange(0, -0.2, -0.002)
    #y1 = np.arange(0, -0.2, -0.002)

    fig, ax = plt.subplots()
    #draw arrows
    Q = ax.quiver(x1, y1, x1, y1, scale=ARROWSIZE)  # draw the arrows

    # gets called with i as the frame number
    def animate(i):
        ax.clear()  # clear the diagram
        ax.contourf(Xf, Yf, func(Xf, Yf))  # draw the function
        ax.set_title('Particle Swarm Optimization')
        ax.set_aspect('equal')
        ax.axis([AXISMIN, AXISMAX, AXISMIN, AXISMAX])

        # draw the velocity arrows (see websites)
        # Q.set
        # https://stackoverflow.com/questions/19329039/plotting-animated-quivers-in-python
        # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.quiver.html

        line, = ax.plot(x1[0:i], y1[0:i], color='blue', lw=1)  # draws the line the point is traveling
        point1, = ax.plot(x1[i], y1[i], marker='X', color='blue')  # draws the moving point

        return point1, line

    # plt.show()

    ani = FuncAnimation(fig, animate, interval=40, blit=False, repeat=True, frames=100)
    ani.save("TLI.gif", dpi=300, writer=PillowWriter(fps=FPS))

    print("Animation saved")


def rosenbrock_function(x, y):
    a = 1
    b = 100
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def rastrigin_function(x, y):
    A = 10
    return A * 2 + x ** 2 - A * np.cos(2 * np.pi * x) + y ** 2 - A * np.cos(2 * np.pi * y)


def quadratic_function(x, y):
    a = 1
    b = 2
    c = 0.5
    d = -1
    e = 1
    f = 0

    return a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y + f


if __name__ == '__main__':
    # positions of particles
    X = [1, 2, 3, 4]
    Y = [5, 2, 8, 1]

    # velocities of particles
    xVel = [1, 2, 3, -2]
    yVel = [2, 5, -3, 0]

    renderPlot(X, Y, xVel, yVel, quadratic_function)
