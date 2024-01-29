from copy import copy
from math import sqrt
from time import time
import os

import multiprocessing
import matplotlib.pyplot as plt

from model.classes.cancerImmuneModel import CancerImmuneModel

from typing import Dict, List, Tuple


"""


Usage:
    To plot an earlier result, run the program like:
        `python paralelRun.py p [FILEPATH]`
    
        
    To run an experiment with default settings, run the program as:
        `python paralelRun.py`

    To run a new experiment, run the program like:
        `python paralelRun.py [RUNS: int] [ITERATIONS: int] [PROCESSES: int] [KWARGS]
            RUNS       - The amount of runs to take average of
            ITERATIONS - The maximum amount of iterations for a single run
            PROCESSES  - The amount of parallel processes to run
            KWARGS     - Keyword arguments (see below)
        or
        `python paralelRun.py [KWARGS]`
            KWARGS     - Keyword arguments (see below)

    Keyword arguments:
        Relevant arguments are presented in a list format, where every list item is associated with the same
        parameter.

        case insensitive:
        ["filename", "fn", "name", "n"]=(int)
        - Filename for the resultFile
        
        ["runs", "r"]=(int)
        - The amount of runs to take average of.
        
        ["maxiter", "iter", "iterations", "i"]=(int)
        - The maximum amount of iterations for a single run.
        
        ["processes", "proc", "p"]=(int)
        - The amount of parallel processes to run.
        
        ["boxl", "boxlen", "bl", "boxlenght"]=(int)
        - The length of the modelled square box.
        
        ["immunefraction", "fractionimmune","immunefrac", "if", "frac", "f"]=(float)
        - The fraction of cells to be occupied by immune cells in the initial state of the model.
        
        case sensitive:
        pImmuneKill=(float)
        - The chance an immune cell kills a cancer cell it occupies the same cell as.

        pCancerMultiply=(float)
        - The chance a cancer cell multiplies.

        pCancerSpawn=(float)
        - The chance a cancer cell spawns on the grid.
    TODO:
    - Dynamic file naming (Manual file naming implemented)
    - Parameter generator function for grid scanning
    - Input arguments (eg. to disable matplotlib figure display for shell automation)
"""


LEN           = 200  # Box lenght
MAX_ITER      = 1000 # Amount of iterations per model run
PROCESSES     = 4    # Amount of paralel processes able to run the model
RUNS          = 8    # Amount of runs to be done
DEF_FILENAME  = "TEMP"

INITIAL_IMMUNE = 0.006 # Initial amount of immune cells (arbitrary in theory)

# Model parameters
PARMS = { # Maybe rework into generator function
        "pImmuneKill" : 0.7, 
        "pCancerMult" : 0.05,
        "pCancerSpawn": 0.01
    }

def paralelRun(maxIter=MAX_ITER, parms = PARMS, runs = RUNS, processes = PROCESSES, boxLen = LEN, immuneFrac = INITIAL_IMMUNE) -> None:
    """
    Parralellized data collection function for CancerImmuneModel

    Measures the occupancy of immune cells as defined by nImmune / boxLength**2 per iteration for a
    set amount of runs. Average, variance and standard deviation are calculated from this and saved
    to ./output/temp.CSV.

    Average and standard deviation are plotted in matplotlib and displayed to the user.
    """
    t_init = time()
    a = multiprocessing.Pool(processes)

    input = [(maxIter, parms, boxLen, immuneFrac) for i in range(runs)]

    print(f"Running {runs} simulations over {processes} processes")
    output = a.starmap(runOnce, input)    
    print(f"Total model runtime: {time() - t_init}")

    t_init = time()
    # output = a.map(par_avgResults, zip(*output)) # Parralellized output processing
    output = avgResults(output)                    # Sequential output processing
    print(f"Total result processing runtime: {time() - t_init}")

    return output


def runOnce(maxIter: int, parms: Dict[str, float], boxLen: int, immuneFrac: float) -> List[float]:
    """
    Runs the model once for the given parameter.
    
    Args:
        maxIter           (int): The amount of iterations to run the model for
        parms (Dict[str,float]): Key-value pairs for the initializer described in .model.cancerImmuneModel
                                 barring length and with.
    
    Returns (List[float]): Immune cell ocupancy relative to the surface of the lattice
    """
    parms = copy(parms)
    model = CancerImmuneModel(boxLen, boxLen, **parms)
    model.seedImmune(round(boxLen**2 * immuneFrac))
    vals = []
    for iter in range(maxIter):
        model.timestep()
        vals.append(model.get_nImmuneCells() / boxLen**2)

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


def saveResults(plot: List[Tuple[float, float, float]], filename: str, parms: Dict[str, float], runs: int, maxIter: int) -> None:
    """
    Saves the calculated results to ./output/TEMP.csv
    
    Args:
        plot (List[Tuple[float, float, float]]): A list of tuples containing the average occupation,
            the variance and the standard deviation for every iteration.
    """
    os.makedirs("output", exist_ok=True)

    with open("output/TEMP.csv", "w") as outFile:
          
        outFile.write(f"{parms}\n")
        outFile.write(f"{runs} runs over {maxIter} iterations\n")
        outFile.write("\n")
        outFile.write("Average, Variance, StdDef\n")
        for (avg, var, stdDef) in plot:
            outFile.write(f"{avg}, {var}, {stdDef}\n")
        outFile.write("\n")
    return


def readData(filename: str) -> List[Tuple[float, float, float]]:
    """
    Reads result datafile and returns it in a format usable by multiplot.
    
    Args:
        filename (str): The filename of the resultfile
    
    Returns (List[Tuple[float, float, float]]): A list of tuples containing the average occupation,
        the variance and the standard deviation for every iteration.

    """
    output = []

    if not ".csv" in filename:
        filename = f"{filename}.csv"

    with open(filename, newline=None) as resultFile:
        line = resultFile.readline().replace("'", '"')
        parms = json.loads(line)

        while line != "\n":
            line = resultFile.readline()
        resultFile.readline()
        line = resultFile.readline()

        while line != None and line != "\n":
            splitline = line.split(',')
            output.append(tuple([float(item) for item in splitline]))
            line = resultFile.readline()
    
    return output


if __name__ == "__main__":
    from sys import argv
    import json

    RUN_SINGLE_PARM = 0
    PLOT = 1

    def main(args: List[str]):
        """
        Main function, see file docstring
        
        Takes argv as input.
        """
        mode, maxIter, parms, runs, processes, boxLen, immuneFrac, filename = parseArgv(args)

        if mode == PLOT:
            multiPlot(readData(filename))
            return
        output = paralelRun(maxIter, parms, runs, processes, boxLen, immuneFrac)
        saveResults(output, filename, parms, runs, maxIter)
        multiPlot(output)

    
    def parseArgv(args: List[str]):
        """
        Argv parser.
        """
        mode       = RUN_SINGLE_PARM
        runs       = RUNS
        maxIter    = MAX_ITER
        filename   = DEF_FILENAME
        processes  = PROCESSES
        boxLen     = LEN
        immuneFrac = INITIAL_IMMUNE
        parms      = PARMS
    
        if len(args) > 1:
            # Plotting past results:
            if args[1].lower() in ['p', 'plot', '1']:
                mode = PLOT
                filename = args[2]

            # Run parallel program
            else:
                args.pop(0)

                if args[0].isnumeric():
                    runs      = int(args.pop(0))
                if args[0].isnumeric():
                    maxIter   = int(args.pop(0))
                if args[0].isnumeric():
                    processes = int(args.pop(0))

                for key, val in [item.split('=') for item in args]:
                    if   key.lower() in ["filename", "fn", "name", "n"]:
                        filename   = val
                    elif key.lower() in ["runs", "r"]:
                        runs       = int(val)
                    elif key.lower() in ["maxiter", "iter", "iterations", "i"]:
                        maxIter    = int(val)
                    elif key.lower() in ["processes", "proc", "p"]:
                        processes  = int(val)
                    elif key.lower() in ["boxl", "boxlen", "bl", "boxlenght"]:
                        boxLen     = int(val)
                    elif key.lower() in ["immunefraction", "fractionimmune","immunefrac", "if", "frac", "f"]:
                        immuneFrac = float(val)
                    else:
                        parms[key] = float(val)

        return mode, maxIter, parms, runs, processes, boxLen, immuneFrac, filename

    main(argv)
