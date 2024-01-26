from copy import copy
from math import sqrt
from time import time
import os

import multiprocessing
import matplotlib.pyplot as plt

from model.classes.cancerImmuneModel import CancerImmuneModel

from typing import Dict, List, Tuple

# TODO: Rework global constants into parameters
LEN = 200
MAX_ITER = 1000
PROCESSES = 8
RUNS      = 8
INTIAL_IMMUNE = LEN**2 * 0.006
PARMS = { # Maybe rework into generator function
        "pImmuneKill" : 0.7, 
        "pCancerMult" : 0.05,
        "pCancerSpawn": 0.01
    }

def main() -> None:
    """
    Parralellized data collection program for CancerImmuneModel
    """
    t_init = time()
    a = multiprocessing.Pool(PROCESSES)

    input = [(MAX_ITER, PARMS) for i in range(RUNS)]

    output = a.starmap(runOnce, input)    
    print(f"Total model runtime: {time() - t_init}")

    t_init = time()
    # output = a.map(par_avgResults, zip(*output)) # Parralellized output processing
    output = avgResults(output)                    # Sequential output processing
    print(f"Total result processing runtime: {time() - t_init}")
    saveResults(output)
    multiPlot(output)

    return


def runOnce(maxIter: int, parms: Dict[str, float]) -> List[float]:
    """
    Runs the model once for the given parameter.
    
    Args:
        maxIter           (int): The amount of iterations to run the model for
        parms (Dict[str,float]): Key-value pairs for the initializer described in .model.cancerImmuneModel
                                 barring length and with.
    
    Returns (List[float]): Immune cell ocupancy relative to the surface of the lattice
    """
    parms = copy(parms)
    model = CancerImmuneModel(LEN, LEN, **parms)
    model.seedImmune(round(INTIAL_IMMUNE))
    vals = []
    for iter in range(maxIter):
        model.timestep()
        vals.append(model.get_nImmuneCells() / LEN**2)

    return vals


def avgResults(plots: List[List[float]]) -> List[Tuple[float, float, float]]:
    """
    Calculates the average, variance and standard deviation for every iteration of every run.

    Args:
        plots (List[List[float]]): List containing the list of Immune cell ocupancy relative to the 
                                   surface of the lattice for every run.
    
    Returns List[Tuple[float, float, float]]: A list of tuples containing the average occupation,
        the variance and the standard deviation for every iteration.

    """
    output: List[Tuple[float]] = []
    for step in zip(*plots):
        average = sum(step) / len(step)

        sumOfSquares = 0
        for point in step:
             sumOfSquares += (point - average)**2
        variance = sumOfSquares / len(step)
        stDev    = sqrt(variance)

        output.append((average, variance, stDev))
    return output


def par_avgResults(step: Tuple[float]) -> Tuple[float, float, float]:
    """
    Implementation of avgResults for use in parrelel pooling.

    Does not seem to lead to significant performance improvement.

    Args:
        step (Tuple[float]): Tuple containing all points at a given iteration.
    
    Returns (Tuple[float, float float]): A tuple containingt he average occupation, the variance and 
        the standard deviation for every iteration.
    """
    average = sum(step) / len(step)

    sumOfSquares = 0
    for point in step:
        sumOfSquares += (point - average)**2
    variance = sumOfSquares / len(step)
    stDev    = sqrt(variance)
    return average, variance, stDev


def multiPlot(plot: List[Tuple[float, float, float]]) -> None:
    """
    Plots the average occupation and standard deviation per iteration.

    Args:
        plot [List[Tuple[float, float, float]]]: A list of tuples containing the average occupation,
        the variance and the standard deviation for every iteration.
    """
    fig, ax = plt.subplots()
    assert isinstance(ax, plt.Axes)
    
    iters = list(range(len(plot)))
    average = [average           for average, variance, stDev in plot]
    hi      = [average + stDev   for average, variance, stDev in plot]
    lo      = [average - stDev   for average, variance, stDev in plot]

    ax.plot(iters, average, label="Average")
    ax.fill_between(iters, lo, hi, label='Standard deviation', alpha=0.5)
    ax.set_ylabel("Immune Cell occupancy")
    ax.set_xlabel("Iteration")
    ax.legend()
    plt.show()


def saveResults(plot: List[Tuple[float, float, float]]) -> None:
    """
    Saves the calculated results to ./output/TEMP.csv
    
    Args:
        plot [List[Tuple[float, float, float]]]: A list of tuples containing the average occupation,
            the variance and the standard deviation for every iteration.
    """
    os.makedirs("output", exist_ok=True)

    with open("output/TEMP.csv", "w") as outFile:
          
        outFile.write(f"{PARMS}\n")
        outFile.write(f"{RUNS} runs over {MAX_ITER} iterations\n")
        outFile.write("\n")
        outFile.write("Average, Variance, StdDef\n")
        for (avg, var, stdDef) in plot:
            outFile.write(f"{avg}, {var}, {stdDef}\n")
        outFile.write("\n")
    return


if __name__ == "__main__":
    main()
    