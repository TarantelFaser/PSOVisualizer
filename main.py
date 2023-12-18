import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

AXISMIN = -10
AXISMAX = 10
DOTS = AXISMAX * 10
FPS = 35
ITERATIONS = 60  # actual iterations of the algorithm
STEPSPERIT = 8  # number of substeps animated between two actual iterations of the algorithm
VMAXFACTOR = 0.1
PARTPERAXIS = 8  # number of particles spread along one axis, total number will be this number squared

# TODO gefundenes globales optimum anzeigen


# takes arrays of coordinates of particles and the velocities
# and a function and renders the plot
# pos array contains a subarray for each particle, containing all the positions
# str is a string that should be printed on the diagrams
def renderPlot(pos, vel, func, str):
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
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.text(7, 11, str, bbox=dict(facecolor='white', alpha=0.75), fontsize=6)

        points = []
        for j in range(len(pos)):  # for each particle
            p = ax.plot(pos[j][i][0], pos[j][i][1], marker='x', color='#b80012')  # draws the moving point
            points.append(p)
        print(f'\r{round((i*100) / ((ITERATIONS - 1) * STEPSPERIT), 1)}%')
        return points

    ani = FuncAnimation(fig, animate, interval=40, blit=False, repeat=True, frames=(ITERATIONS - 1) * STEPSPERIT)
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


def standardParticleSwarmOptimization(func, cC=1.49, cS=1.49, w=0.5):
    particles = []  # in the first element is the positional data of all the particles in the first iteration
    velocities = []
    personalBests = []

    w = 0.5 + (random.random() / 2)

    # append arrays for first iteration
    particles.append([])
    velocities.append([])

    # initialize positions
    step = (np.abs(AXISMIN) + np.abs(AXISMAX)) / PARTPERAXIS
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
            cogComp = [cC * r1 * (personalBests[i][0] - lastPos[0]),
                       cC * r1 * (personalBests[i][1] - lastPos[1])]
            socComp = [cS * r2 * (globalBest[0] - lastPos[0]),
                       cS * r2 * (globalBest[1] - lastPos[1])]
            newVel = [origVel[0] + cogComp[0] + socComp[0],
                      origVel[1] + cogComp[1] + socComp[1]]

            velocities[t].append(getClampedVel(newVel))

            # compute new positions
            particles[t].append(
                [particles[t - 1][i][0] + velocities[t][i][0],
                 particles[t - 1][i][1] + velocities[t][i][1]]
            )

            # update personal and global bests
            if func(particles[t][i][0], particles[t][i][1]) < func(personalBests[i][0], personalBests[i][1]):
                personalBests[i] = particles[t][i]
            if func(particles[t][i][0], particles[t][i][1]) < func(globalBest[0], globalBest[1]):
                globalBest = particles[t][i]

        t = t + 1

    textstr = "inertiaWeight = " + str(round(w,2)) + "\ncognitiveComp = " + str(round(cC,2)) + "\nsocComp = " + str(round(cS,2))
    return particles, velocities, textstr


def stepsBetweenPositions(particles):
    smooth = []

    for i in range(len(particles[0])):  # for each particle
        smooth.append([])
        smooth[i].append(particles[0][i])  # add the very first position to the smoothSteps array
        for j in range(ITERATIONS - 1):
            currentPos = particles[j][i]
            nextPos = particles[j + 1][i]
            smoothVectors = np.linspace(currentPos, nextPos, STEPSPERIT + 1)
            for s in smoothVectors[1:]:
                smooth[i].append(s)

    return smooth


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
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (c_x * x ** 2 + c_y * y ** 2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * c_x * x) + np.cos(2 * np.pi * c_y * y)))
    term3 = np.exp(1) + 20
    return term1 + term2 + term3


def himmelblau_function(x, y):
    term1 = (x ** 2 + y - 11) ** 2
    term2 = (x + y ** 2 - 7) ** 2
    return term1 + term2


def other_function(x, y):
    return np.sin(x) + np.cos(y)


def generate_steps_between_vectors(start_vector, end_vector, num_steps):
    steps = np.linspace(start_vector, end_vector, num_steps)
    return steps


if __name__ == '__main__':
    # selectedFunc = rastrigin_function
    # selectedFunc = rosenbrock_function
    # selectedFunc = quadratic_function
    # selectedFunc = ackley_function
    # selectedFunc = himmelblau_function
    selectedFunc = other_function
    pos, vel, text = standardParticleSwarmOptimization(selectedFunc)
    smoothSteps = stepsBetweenPositions(pos)  # adds extra frames to make the animation more smooth
    print("done computing, starting rendering now")
    renderPlot(smoothSteps, vel, selectedFunc, text)
