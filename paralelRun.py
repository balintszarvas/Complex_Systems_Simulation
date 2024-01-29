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
        "pImmuneKill" : [0.5],
        "pCancerMult" : [0.4, 0.5, 0.6],
        "pCancerSpawn": [0.1],
        }

def parScan():
    """Loop through possible parameters"""
    for cancermulti in PARMS["pCancerMult"]:
        for immunekill in PARMS["pImmuneKill"]:
            for cancerspawn in PARMS["pCancerSpawn"]:
                for initial_imm in INITIAL_IMMUNE:
                    current_parms = {
                        "pCancerMult": cancermulti,
                        "pImmuneKill": immunekill,
                        "pCancerSpawn": cancerspawn
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
    output = avgResults(output)
    print(f"Total result processing runtime: {time() - t_init}")

    return output


def runOnce(maxIter: int, parms: Dict[str, float], boxLen: int, immuneFrac: float) -> List[float]:
    """
    Runs the model once for the given parameter.
    
    Args:
        maxIter (int): The amount of iterations to run the model for.
        INTIAL_IMMUNE (float): Initial amount of immune cells.
        parms (Dict[str, float]): Model parameters (pCancerMult, pImmuneKill, pCancerSpawn).

    Returns:
        Dict[str, List[float]]: A dictionary with keys "Immune" and "Cancer", each containing a list of cell occupancy values.
    """
    parms = copy(parms)
    model = CancerImmuneModel(boxLen, boxLen, **parms)
    model.seedImmune(round(boxLen**2 * immuneFrac))
    vals = {"Immune": [], "Cancer": []}
    for iter in range(maxIter):
        model.timestep()
        vals["Immune"].append(model.get_nImmuneCells() / boxLen**2)
        vals["Cancer"].append(model.get_nCancerCells() / boxLen**2)

    return vals


def calculate_statistics(values: List[float]) -> Tuple[float, float, float]:
    """
    Calculates statistical measures for a given list of values.

    Args:
        values (List[float]): List of numerical values.

    Returns:
        Tuple[float, float, float]: A tuple containing the average, variance, and standard deviation of the input values.
    """
    average = sum(values) / len(values)
    variance = sum((x - average) ** 2 for x in values) / len(values)
    st_dev = sqrt(variance)
    return average, variance, st_dev

def avgResults(plots: List[Dict[str, List[float]]]) -> Dict[str, List[Tuple[float, float, float]]]:
    """
    Computes average, variance, and standard deviation for each iteration of model runs.

    Args:
        plots (List[Dict[str, List[float]]]): List of dictionaries containing model results for "Immune" and "Cancer" cell types.

    Returns:
        Dict[str, List[Tuple[float, float, float]]]: A dictionary with keys "Immune" and "Cancer", each containing a list of tuples (average, variance, standard deviation).
    """
    results = {}
    for key in plots[0].keys():
        key_values = [plot[key] for plot in plots]
        results[key] = [calculate_statistics(step) for step in zip(*key_values)]
    return results


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


def multiPlot(results: List[Dict[str, List[Tuple[float, float, float]]]]) -> None:
    """
    Plots the results of model simulations.

    Args:
        plot (List[Dict[str, List[Tuple[float, float, float]]]]): The calculated statistics for immune and cancer cell occupancy.

    Returns:
        None: This function does not return anything but plots the results.
    """
    fig, ax = plt.subplots()

    iters = range(len(results))
    averages = [avg for avg, _, _ in results]
    std_devs = [std_dev for _, _, std_dev in results]

    ax.plot(iters, averages, label=f"Average")
    ax.fill_between(iters, 
                        [a - s for a, s in zip(averages, std_devs)],
                        [a + s for a, s in zip(averages, std_devs)],
                        alpha=0.5, label=f"Std Dev")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cell occupancy")
    ax.legend()

    plt.tight_layout()
    plt.show()


def saveResults(plot: List[Tuple[float, float, float]], filename: str, parms: Dict[str, float], runs: int, maxIter: int) -> None:
    """
    Saves the calculated results to a CSV file.

    Args:
        plot (List[Tuple[float, float, float]]): A list of tuples containing the average occupation,
            the variance and the standard deviation for every iteration.
    """
    os.makedirs("Output", exist_ok=True)

    with open("output/TEMP.csv", "w") as outFile:
          
        outFile.write(f"{parms}\n")
        outFile.write(f"{runs} runs over {maxIter} iterations\n")
        outFile.write("\n")
        outFile.write("AverageImmune, VarianceImmune, StdDefImmune, AverageCancer, VarianceCancer, StdDefCancer\n")
        for CellType in plot.keys():
            for (avg, var, stdDef) in plot[CellType]:
                outFile.write(f"{avg}, {var}, {stdDef}")
            outFile.write('\n')
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
