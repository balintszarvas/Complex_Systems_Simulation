from typing import Tuple, List
from paralelRun import readData
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.optimize as spopt
"""
TODO: 
    - document args
    - Implement command line arguments
d/dt Immune   = a * bacteria * immune - b * immune^2
d/dt Bacteria = c * bacteria^2 - a * bacteria * immune  + spawnRate

a -> killRate
b -> imm overcrowding
c -> bac growthrate
d -> imm killRate

d -> may actually be alpha
"""
ALPHA = 0
BETA  = 1
GAMMA = 2
SPAWN = 3
DELTA = 4

X_GUESS = np.array([0.5, 0.25, 0.4, 0.1])
DT = 0.1
FILENAME = "output\\'pImmuneKill' 0.5, 'pCancerMult' 0.4, 'pCancerSpawn' 0.1.csv"

attempts = 0

def d_dtImmuneCells(alpha: float, beta: float, gamma: float, delta: float, bacteria: float, immune: float, 
                    spawnRate: float) -> float:
    """
    Partial differetntial equation for immune cells
    """

    return (alpha * bacteria * immune) - (beta * immune**2)


def d_dtBacteria(alpha: float, beta: float, gamma: float, delta:float, bacteria:float, immune:float, 
                 spawnRate:float) -> float:
    """
    Partial differetntial equation for bacterial cells
    """
    return (gamma * bacteria**2) - (alpha * immune * bacteria) + spawnRate


def rungeKuttaTimestep(dt:float, alpha:float, beta:float, gamma:float, delta:float, bacteria:float, 
                       immune:float, spawnRate:float) -> Tuple[float, float]:
    """
    Implementation of Runge-kutta method for numerical integration.

    Returns (Tuple[float, float]): The next values for immune and bacterial cells respectively.

    Runge Kutta algorithm adapted from: 
    Introduction to Computational science - Homework Assignment 1, as submitted by Hajo Groen.
    """
    output = []
    kI = [dt * d_dtImmuneCells(alpha, beta, gamma, delta, bacteria, immune, spawnRate)]
    kB = [dt * d_dtBacteria   (alpha, beta, gamma, delta, bacteria, immune, spawnRate)]

    for i in range(3):
        knI = dt * d_dtImmuneCells(alpha, beta, gamma, delta, bacteria + (kB[-1] / 2), immune  + (kI[-1] / 2), spawnRate)
        knB = dt * d_dtBacteria   (alpha, beta, gamma, delta, bacteria + (kB[-1] / 2), immune  + (kI[-1] / 2), spawnRate)
        if knB > 1.0:
            knB = 1.0
        kI.append(knI)
        kB.append(knB)

    for kList in [kI, kB]:
        integrand = (kList[0] + (2*kList[1]) + (2*kList[2]) + kList[3]) / 6
        output.append(integrand)
    
    return tuple(output)


def runODE(tMax:float, dt:float, alpha:float, beta:float, gamma:float, delta:float, bacteriaInit:float, 
           immuneInit:float, spawnRate:float, wholeSteps = True) -> Tuple[List[float],List[float]]:
    """
    Runs the ODE using the Runge-kutta method for numerical integration with timesteps dt until its bigger than tMax.
    """
    iters = 1
    interval = round(1 / dt)
    output   = ([immuneInit],[bacteriaInit])
    immune   = immuneInit
    bacteria = bacteriaInit

    while len(output[0]) < tMax:
        dimmune, dbacteria = rungeKuttaTimestep(dt, alpha, beta, gamma,delta, bacteria, immune, spawnRate)

        immune   += dimmune
        bacteria += dbacteria
        if bacteria > 1.0:
            bacteria = 1.0

        if iters % interval == 0:
            output[0].append(immune)
            output[1].append(bacteria)
        
        iters += 1
    
    return output


def toMinimize(x: np.array, dt, immune: List[float], bacteria: List[float]):
    """
    Function to minimize for 
    """
    global attempts
    if not attempts % 100:
        print(f"attempt {attempts}")
    attempts += 1

    tMax = len(immune)
    ODE_immune, ODE_bacteria = runODE(tMax, dt, x[ALPHA], x[BETA], x[GAMMA], 0, bacteria[0], immune[0], x[SPAWN])
    score  = sum([(spatial - ode)**2 for spatial, ode in zip(immune  , ODE_immune  )])
    score += sum([(spatial - ode)**2 for spatial, ode in zip(bacteria, ODE_bacteria)])

    return score

def fitODE(immuneCells, bacterialCells, xGuess, dt = DT) -> np.ndarray:
    return spopt.minimize(toMinimize, xGuess, (dt, immuneCells, bacterialCells))


def main(filename = FILENAME, xGuess: np.ndarray = X_GUESS):
    data = readData(filename)
    immuneCells    = [immune   for immune, _, _, bacteria, _, _ in data]
    bacterialCells = [bacteria for immune, _, _, bacteria, _, _ in data]

    xOut = fitODE(immuneCells, bacterialCells, xGuess)["x"]
    print(f"xOut = {xOut}")
    ODE_immune, ODE_bacteria = runODE(len(immuneCells), DT, xOut[ALPHA], xOut[BETA], xOut[GAMMA], 0, 
                                    bacterialCells[0], immuneCells[0], xOut[SPAWN])
    
    plotComparison(immuneCells, bacterialCells, ODE_immune, ODE_bacteria)
    

def plotComparison(immuneSpatial: List[float], BacteriaSpatial: List[float], ODE_immune: List[float], 
                   ODE_bacteria: List[float]) -> None:
    fig, ax = plt.subplots()
    assert(isinstance(ax, plt.Axes))
    ax.plot(ODE_immune     , color="tab:blue"  , label = "Immune Fitted ODE")
    ax.plot(immuneSpatial  , color="tab:blue"  , label = "Immune Spatial Model"  , linestyle="--")
    ax.plot(ODE_bacteria   , color="tab:orange", label = "Bacteria Fitted ODE")
    ax.plot(BacteriaSpatial, color="tab:orange", label = "Bacteria Spatial model", linestyle="--")

    ax.set_ylabel("Cell occupancy")
    ax.set_xlabel("Time")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
