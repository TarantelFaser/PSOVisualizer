import math
import random

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

AXISMIN = -10
AXISMAX = 10
DOTS = 100
ARROWSIZE = 25
FPS = 25
ITERATIONS = 100


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

    # arrange returns array of evenly spaced values (this could be replaced by arrays of positions from the pso algorithm)
    x1 = np.arange(0, -2, -0.02)
    y1 = x1 ** 2
    #x1 = np.arange(0, -0.2, -0.002)
    #y1 = np.arange(0, -0.2, -0.002)

    # gets called with i as the frame number
    def animate(i):
        ax.clear()  # clear the diagram
        ax.contourf(Xf, Yf, func(Xf, Yf))  # draw the function
        ax.set_title('Particle Swarm Optimization')
        ax.set_aspect('equal')
        ax.axis([AXISMIN, AXISMAX, AXISMIN, AXISMAX])

        #line, = ax.plot(x1[0:i], y1[0:i], color='red', lw=1)  # draws the line the point is traveling
        #point1, = ax.plot(x1[i], y1[i], marker='X', color='red')  # draws the moving point

        points = []
        for j in range(0, len(pos[0])):
            p = ax.plot(pos[i][j][0], pos[i][j][1], marker='X', color='red')  # draws the moving point
            points.append(p)

        return points

    ani = FuncAnimation(fig, animate, interval=40, blit=False, repeat=True, frames=ITERATIONS)
    ani.save("TLI.gif", dpi=300, writer=PillowWriter(fps=FPS))
    print("Animation saved")

def standardParticleSwarmOptimization(swarmSize, func, c1 = 0.1, c2 = 0.1, w = 0.1):
    particles = [] # in the first element is the positional data of all the particles in the first iteration
    velocities = []
    personalBests = []
    globalBest = []

    # append arrays for first iteration
    particles.append([])
    velocities.append([])

    # initialize positions
    for i in range(swarmSize):
        particles[0].append([0, i])
        velocities[0].append([0, 0])
        personalBests.append(particles[0][i])

    # initialize the global best to the value of the first particle
    globalBest = particles[0][0]

    # update the global best to the best position of a particle
    for i in range(swarmSize):
        if func(particles[0][i][0], particles[0][i][1]) < func(globalBest[0], globalBest[1]):
            globalBest = particles[0][i]

    t = 1
    while t < ITERATIONS:
        particles.append([])
        velocities.append([])
        for i in range(swarmSize): #foreach particle
            #compute new velocities
            r1 = random.random()
            r2 = random.random()
            origVel = [w * velocities[t-1][i][0], w * velocities[t-1][i][1]]
            cogComp = [c1*r1*(personalBests[i][0]-particles[t-1][i][0]), c1*r1*(personalBests[i][1]-particles[t-1][i][1])]
            socComp = [c2*r2*(globalBest[0] - particles[t-1][i][0]), c2*r2*(globalBest[1] - particles[t-1][i][1])]
            velocities[t].append([origVel[0] + cogComp[0] + socComp[0], origVel[1] + cogComp[1] + socComp[1]])

            #compute new positions
            particles[t].append([particles[t-1][i][1] + velocities[t][i][0], particles[t-1][i][1] + velocities[t][i][1]])

            #update personal and global bests
            if func(particles[t][i][0], particles[t][i][1]) < func(personalBests[i][0], personalBests[i][1]):
                personalBests[i] = particles[t][i]
            if func(particles[t][i][0], particles[t][i][1]) < func(globalBest[0], globalBest[1]):
                globalBest = particles[t][i]

        t = t + 1

    return particles, velocities

    #TODO velocity clamping is needed asap!


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

    pos, vel = standardParticleSwarmOptimization(5, quadratic_function, 5)
    print("done computing, starting rendering now")
    renderPlot(pos, vel, quadratic_function)
