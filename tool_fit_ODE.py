
from typing import Tuple, List
from tool_paralelRun import readData
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.optimize as spopt

"""ODE Parameter fitting for spatial bacteriaImmuneModel.
Fits input guess parameters to data from spatial ODE and prints fitted parameters to console.

Partial Differential Equations:
d/dt Immune   = a * bacteria * immune - b * immune^2
d/dt Bacteria = c * bacteria^2 - a * bacteria * immune  + spawn

a     -> immunecell killRate
b     -> immunecell overcrowding
c     -> bacterial growthrate
spawn -> Bacterial spawnrate

Usage:
    `python ./tool_fit_ODE [filename] [xGuess] [plotGuess]`

    filename : Filename or filepath to resultfile relative from active user folder.
    xGuess   : Guess parameters for ODE fit, formatted like stock python list  of four elements
               between quotes.
               Example: `"[0.1, 0.1, 0.1, 0.1]"`
    plotGuess: Bool like string of whether the input guess should be plotted to the input file data 
               without fitting.
               Example: `True`, `False`
"""

ALPHA = 0
BETA  = 1
GAMMA = 2
SPAWN = 3

X_GUESS = np.array([0.5, 0.25, 0.4, 0.1])
DT = 0.1
FILENAME = "data\\temp"

attempts = 0

def d_dtImmuneCells(alpha: float, beta: float, gamma: float, bacteria: float, immune: float, spawn: float) -> float:
    """
    Partial differential equation for immune cells.

    Args:
        alpha    (float): ODE killRate.
        beta     (float): ODE Immune Overcrowding.
        gamma    (float): ODE Bacterial growthrate (Unused but present for parity).
        bacteria (float): Fraction of possible bacteria cells present.
        immune   (float): Fraction of possible immune cells present.
        spawn    (float): ODE bacteria spawnrate (Unused but present for parity).
    """

    return (alpha * bacteria * immune) - (beta * immune**2)


def d_dtBacteria(alpha: float, beta: float, gamma: float, bacteria:float, immune:float, spawn:float) -> float:
    """
    Partial differetntial equation for bacterial cells

    Args:
        alpha    (float): ODE killRate.
        beta     (float): ODE Immune Overcrowding (Unused but present for parity).
        gamma    (float): ODE Bacterial growthrate.
        bacteria (float): Fraction of possible bacteria cells present.
        immune   (float): Fraction of possible immune cells present.
        spawn    (float): ODE bacteria spawnrate.
    """
    return (gamma * bacteria**2) - (alpha * immune * bacteria) + spawn


def rungeKuttaTimestep(dt:float, alpha:float, beta:float, gamma:float, bacteria:float, 
                       immune:float, spawn:float) -> Tuple[float, float]:
    """
    Implementation of Runge-kutta method for numerical integration.

    Args:
        dt       (float): Timestep for runge-kutta algorithm
        alpha    (float): ODE killRate.
        beta     (float): ODE Immune Overcrowding (Unused but present for parity).
        gamma    (float): ODE Bacterial growthrate.
        bacteria (float): Fraction of possible bacteria cells present.
        immune   (float): Fraction of possible immune cells present.
        spawn    (float): ODE bacteria spawnrate.

    Returns (Tuple[float, float]): The next values for immune and bacterial cells respectively.

    Runge Kutta algorithm adapted from: 
    Introduction to Computational science - Homework Assignment 1, as submitted by Hajo Groen.
    """
    output = []
    kI = [dt * d_dtImmuneCells(alpha, beta, gamma, bacteria, immune, spawn)]
    kB = [dt * d_dtBacteria   (alpha, beta, gamma, bacteria, immune, spawn)]

    for i in range(3):
        knI = dt * d_dtImmuneCells(alpha, beta, gamma, bacteria + (kB[-1] / 2), immune  + (kI[-1] / 2), spawn)
        knB = dt * d_dtBacteria   (alpha, beta, gamma, bacteria + (kB[-1] / 2), immune  + (kI[-1] / 2), spawn)
    
        kI.append(knI)
        kB.append(knB)

    for kList in [kI, kB]:
        integrand = (kList[0] + (2*kList[1]) + (2*kList[2]) + kList[3]) / 6
        output.append(integrand)
    
    return tuple(output)


def runODE(tMax:float, dt:float, alpha:float, beta:float, gamma:float, bacteriaInit:float, 
           immuneInit:float, spawn:float) -> Tuple[List[float],List[float]]:
    """
    Runs the ODE using the Runge-kutta method for numerical integration with timesteps dt until its bigger than tMax.

    Args:
        tMax     (float): Maximum time for ODE run
        dt       (float): Timestep for runge-kutta algorithm
        alpha    (float): ODE killRate.
        beta     (float): ODE Immune Overcrowding (Unused but present for parity).
        gamma    (float): ODE Bacterial growthrate.
        bacteria (float): Fraction of possible bacteria cells present.
        immune   (float): Fraction of possible immune cells present.
        spawn    (float): ODE bacteria spawnrate.
    """
    iters = 1
    interval = round(1 / dt)
    output   = ([immuneInit],[bacteriaInit])
    immune   = immuneInit
    bacteria = bacteriaInit

    while len(output[0]) < tMax:
        dimmune, dbacteria = rungeKuttaTimestep(dt, alpha, beta, gamma, bacteria, immune, spawn)

        immune   += dimmune
        bacteria += dbacteria
        if bacteria > 1.0:
            bacteria = 1.0

        if iters % interval == 0:
            output[0].append(immune)
            output[1].append(bacteria)
        
        iters += 1
    
    return output


def toMinimize(x: "np.ndarray[float]", dt: float, immune: List[float], bacteria: List[float]) -> float:
    """
    Function to minimize for scypy.opt.minimize

    Args:
        X  (np.ndarray[float]): Array containing the ODE parameters in the following order:
            [alpha, beta, gamma, spawn]
        dt             (float): Timestep for runge-kutta algorithm
        immune   (List[float]): List of Immune cell occupation per iteration of spatial model run.
        bacteria (List[float]): List of bacteria cell occupation per iteration of spatial model run.
    
    Returns (float): The sum of the sum of least squares for the difference between the ODE and spatial model.
    """
    global attempts
    if not attempts % 100:
        print(f"attempt {attempts}")
    attempts += 1

    tMax = len(immune)
    ODE_immune, ODE_bacteria = runODE(tMax, dt, x[ALPHA], x[BETA], x[GAMMA], bacteria[0], immune[0], x[SPAWN])
    score  = sum([(spatial - ode)**2 for spatial, ode in zip(immune  , ODE_immune  )])
    score += sum([(spatial - ode)**2 for spatial, ode in zip(bacteria, ODE_bacteria)])

    return score

def fitODE(immune: List[float], bacteria: List[float], xGuess: "np.ndarray[float]", dt = DT) -> "np.ndarray[float]":
    """
    Fits ODE parameters to cell occupations per iteration of spatial model run.

    Args:
        immune       (List[float]): List of Immune cell occupation per iteration of spatial model run.
        bacteria     (List[float]): List of bacteria cell occupation per iteration of spatial model run.
        xGuess (np.ndarray[float]): Array containing the initial guess for ODE parameters in the following 
                                    order: [alpha, beta, gamma, spawn].
        dt                 (float): Runge-kutta timestep.

    Returns (np.ndarray[float]): The fitted ODE parameters in the same order as xGuess.
    """

    return spopt.minimize(toMinimize, xGuess, (dt, immune, bacteria))


def loadData(filename: str) -> Tuple[List[float], List[float]]:
    """
    Loads relevant immunecell and bacterial cell data from previously performed spatial model run.

    Args:
        filename (str): Filename/ path of the spatial output file to be read from.
    
    Returns (Tuple[List[float], List[float]]): A tuple containing average Immune cell occupancy per 
        iteration and average bacterial cell occupation per iteration, respectively.
    """
    data = readData(filename)
    immuneCells    = [immune   for immune, _, _, bacteria, _, _ in data]
    bacterialCells = [bacteria for immune, _, _, bacteria, _, _ in data]
    return immuneCells, bacterialCells


def main(filename = FILENAME, xGuess: "np.ndarray[float]"= X_GUESS, plotGuess: bool = False) -> None:
    """
    Fits input guess parameters to data from spatial ODE and prints fitted parameters to console.

    Args: 
        filename             (str): Filename/ path of the spatial output file to be read from.
        xGuess (np.ndarray[float]): Array containing the initial guess for ODE parameters in the following 
                                    order: [alpha, beta, gamma, spawn].
        plotGuess          (float): True if the input guess parameters must be plotted to the spatial 
                                    file without fitting, else False.
    """
    immuneCells, bacterialCells = loadData(filename)

    if plotGuess:
        xOut = xGuess
    else:
        out = fitODE(immuneCells, bacterialCells, xGuess)
        xOut = out["x"]
        print(out["message"])
        print(f"xOut = {xOut}")
        print(f"xOut = [{','.join([str(item) for item in xOut])}]")
    ODE_immune, ODE_bacteria = runODE(len(immuneCells), DT, xOut[ALPHA], xOut[BETA], xOut[GAMMA], 0, 
                                    bacterialCells[0], immuneCells[0], xOut[SPAWN])
    
    plotComparison(immuneCells, bacterialCells, ODE_immune, ODE_bacteria)
    

def plotComparison(immuneSpatial: List[float], BacteriaSpatial: List[float], ODE_immune: List[float], 
                   ODE_bacteria: List[float]) -> None:
    """
    Plots spatial immunecell and bacteriacell data with fitted ODE data
    
    Args:
        immuneSpatial   (List[float]): List of immune cell occupancy per iteration for spatial model.
        BacteriaSpatial (List[float]): List of bacteria cell occupancy per iteration for spatial model.
        ODE_immune      (List[float]): List of immune cell popuilation per timestep of ODE model.
        ODE_bacteria    (List[float]): List of bacteria cell popuilation per timestep of ODE model.
    """
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
    from sys import argv
    import json
    filename = FILENAME
    xGuess   = X_GUESS
    plotGuess = False
    if len(argv) > 1:
        filename = argv[1]
    if len(argv) > 2:
        xGuess = np.array(json.loads(argv[2]))
    if len(argv) > 3:
        plotGuess = bool(argv[3])

    main(filename, xGuess, plotGuess)