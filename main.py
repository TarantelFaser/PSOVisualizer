import math
import random

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

AXISMIN = -10
AXISMAX = 10
DOTS = AXISMAX * 10
FPS = 25
ITERATIONS = 100
VMAXFACTOR = 0.1


# takes arrays of coordinates of particles and the velocities
# and a function and renders the plot
def renderPlot(pos, vel, func):
    fig, ax = plt.subplots()
    plt.xlabel('X')
    plt.ylabel('Y')

    # get the contour plot for the function (func : R^2 -> R)
    xf = np.linspace(AXISMIN, AXISMAX, DOTS)
    yf = np.linspace(AXISMIN, AXISMAX, DOTS)
    Xf, Yf = np.meshgrid(xf, yf)

    # gets called with i as the frame number
    def animate(i):
        ax.clear()  # clear the diagram
        ax.contourf(Xf, Yf, func(Xf, Yf))  # draw the function
        ax.set_title('Particle Swarm Optimization')
        ax.set_aspect('equal')
        ax.axis([AXISMIN, AXISMAX, AXISMIN, AXISMAX])

        # line, = ax.plot(x1[0:i], y1[0:i], color='red', lw=1)  # draws the line the point is traveling
        # point1, = ax.plot(x1[i], y1[i], marker='X', color='red')  # draws the moving point

        points = []
        for j in range(0, len(pos[0])):
            # xs = np.arange(pos[i][j][0])
            # for k in range()
            p = ax.plot(pos[i][j][0], pos[i][j][1], marker='X', color='red')  # draws the moving point
            points.append(p)

        return points

    ani = FuncAnimation(fig, animate, interval=40, blit=False, repeat=True, frames=ITERATIONS)
    ani.save("TLI.gif", dpi=300, writer=PillowWriter(fps=FPS))
    print("Animation saved")


def getLengthOfVector(ar):
    return np.sqrt(ar[0] ** 2 + ar[1] ** 2)


def getClampedVel(newVel, delta=VMAXFACTOR):
    vMaxValue = delta * (np.abs(AXISMAX) + np.abs(AXISMIN))  # max allowed velocity
    newVelValue = getLengthOfVector(newVel)  # value / betrag of new calculated velocity

    if newVelValue >= vMaxValue:
        normalisedNewVel = [newVel[0] / newVelValue,
                            newVel[1] / newVelValue]  # normalised vector of new calculated velocity
        return [vMaxValue * normalisedNewVel[0], vMaxValue * normalisedNewVel[1]]
    else:
        return newVel


def standardParticleSwarmOptimization(func, c1=0.1, c2=0.1, w=0.1):
    particles = []  # in the first element is the positional data of all the particles in the first iteration
    velocities = []
    personalBests = []

    # append arrays for first iteration
    particles.append([])
    velocities.append([])

    # initialize positions
    step = (np.abs(AXISMIN) + np.abs(AXISMAX)) / 8
    x, y = np.arange(AXISMIN, AXISMAX, step), np.arange(AXISMIN, AXISMAX, step)
    partCount = 0
    for a in x:
        for b in y:
            particles[0].append([a, b])
            velocities[0].append([0, 0])
            personalBests.append(particles[0][partCount])
            partCount = partCount + 1

    # initialize the global best to the value of the first particle
    globalBest = particles[0][0]

    # update the global best to the best position of a particle
    for i in range(partCount):
        if np.all(func(particles[0][i][0], particles[0][i][1]) < func(globalBest[0], globalBest[1])):
            globalBest = particles[0][i]

    t = 1
    while t < ITERATIONS:
        particles.append([])
        velocities.append([])
        for i in range(partCount):  # foreach particle
            lastPos = particles[t - 1][i]

            # compute new velocities
            r1 = random.random()
            r2 = random.random()
            origVel = [w * velocities[t - 1][i][0],
                       w * velocities[t - 1][i][1]]
            cogComp = [c1 * r1 * (personalBests[i][0] - lastPos[0]),
                       c1 * r1 * (personalBests[i][1] - lastPos[1])]
            socComp = [c2 * r2 * (globalBest[0] - lastPos[0]),
                       c2 * r2 * (globalBest[1] - lastPos[1])]
            newVel = [origVel[0] + cogComp[0] + socComp[0],
                      origVel[1] + cogComp[1] + socComp[1]]

            velocities[t].append(getClampedVel(newVel))

            # compute new positions
            particles[t].append(
                [particles[t - 1][i][1] + velocities[t][i][0], particles[t - 1][i][1] + velocities[t][i][1]])

            # update personal and global bests
            if func(particles[t][i][0], particles[t][i][1]) < func(personalBests[i][0], personalBests[i][1]):
                personalBests[i] = particles[t][i]
            if func(particles[t][i][0], particles[t][i][1]) < func(globalBest[0], globalBest[1]):
                globalBest = particles[t][i]

        t = t + 1

    return particles, velocities


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


def ackley_function(x, y):
    c_x = 1.5  # Coefficient for the x term
    c_y = 0.8  # Coefficient for the y term
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (c_x * x**2 + c_y * y**2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * c_x * x) + np.cos(2 * np.pi * c_y * y)))
    term3 = np.exp(1) + 20
    return term1 + term2 + term3


def himmelblau_function(x, y):
    term1 = (x**2 + y - 11)**2
    term2 = (x + y**2 - 7)**2
    return term1 + term2

def other_function(x,y):
    return np.sin(x) + np.cos(y)


if __name__ == '__main__':
    # positions of particles
    X = [1, 2, 3, 4]
    Y = [5, 2, 8, 1]

    # velocities of particles
    xVel = [1, 2, 3, -2]
    yVel = [2, 5, -3, 0]

    # selectedFunc = rastrigin_function
    # selectedFunc = rosenbrock_function
    # selectedFunc = quadratic_function
    # selectedFunc = ackley_function
    # selectedFunc = himmelblau_function
    selectedFunc = other_function
    pos, vel = standardParticleSwarmOptimization(selectedFunc)
    print("done computing, starting rendering now")
    renderPlot(pos, vel, selectedFunc)
