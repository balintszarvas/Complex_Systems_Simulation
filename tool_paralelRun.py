
from copy import copy
from math import sqrt
from time import time, sleep
import os
import json

import multiprocessing
from multiprocessing import Pool
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

from model.classes.bacteriaImmuneModel import bacteriaImmuneModel

from typing import Dict, List, Tuple


"""Parallel running program for Bacteria immune model.
Usage:
    To plot an earlier result, run the program like:
        `python tools_paralelRun.py p [FILEPATH] [PLOTNAME: optional]`
            FILEPATH   - Path of the exported CSV file.
            PLOTNAME   - Title for the plotted image

    To run an experiment with default settings, run the program as:
        `python tools_paralelRun.py`

    To run a new experiment, run the program like:
        `python tools_paralelRun.py [RUNS: int] [ITERATIONS: int] [PROCESSES: int] [KWARGS]
            RUNS       - The amount of runs to take average of
            ITERATIONS - The maximum amount of iterations for a single run
            PROCESSES  - The amount of parallel processes to run
            KWARGS     - Keyword arguments (see below)
        or
        `python paralelRun.py [KWARGS]`
            KWARGS     - Keyword arguments (see below)

    To run sensitivity analyses, run the run the program like:
        `python tools_paralelRun.py [RUNS: int] [ITERATIONS: int] [PROCESSES: int]  sensitivity'
            RUNS       - The amount of runs to take average of
            ITERATIONS - The maximum amount of iterations for a single run (use int > 600 for equilibrium)
            PROCESSES  - The amount of parallel processes to run (recommended to set value to number of cores on pc)

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

        ["noplot"]=(bool)
        - disable plotting of results afterwards

        case sensitive model parameters:
            pImmuneKill=(float/list[float])
            - The chance an immune cell kills a bacteria cell it occupies the same cell as.

            pBacteriaMultiply=(float/list[float])
            - The chance a bacteria cell multiplies.

            pBacteriaSpawn=(float/list[float])
            - The chance a bacteria cell spawns on the grid.

            A list structure can also be entered, which makes the program loop through all possible
            combinations of these parameters.
"""


LEN           = 200  # Box lenght
MAX_ITER      = 1000 # Amount of iterations per model run
PROCESSES     = 4    # Amount of paralel processes able to run the model
RUNS          = 4    # Amount of runs to be done
DEF_FILENAME  = "TEMP"

INITIAL_IMMUNE = 0.006 # Initial amount of immune cells (arbitrary in theory)

# Model parameters
PARMS = { # Maybe rework into generator function
        "pImmuneKill" : 0.5,
        "pBacteriaMult" : 0.4,
        "pBacteriaSpawn": 0.1
        }


RUN_SINGLE_PARM = 0
PLOT = 1
PARAMETER_SCAN = 2


def plotEquilibriumAnalysis(sensitivity_results,equilibrium_range):
    """
    Plot interactions between parameters for immune and bacteria cells at equilibrium

    Parameters:
    - sensitivity_results (dict):   Dictionary containing sensitivity analysis results
    - equilibrium_range (int)   :   Number of last iterations to calculate equilibrium average

    Returns:
    Saves and shows plot
    """
    for param, results in sensitivity_results.items():
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,6))

        equilibrium_vals_bacteria =[]
        equilibrium_vals_immune =[]
        param_vals =[]

        for val, data in results:
            # calculate the equilibrium average for the last `equilibrium_range`
            equilibrium_avg_bacteria = np.mean([avgB for _, _, _, avgB, _,_ in data[-equilibrium_range:]])
            equilibrium_avg_immune = np.mean([avgI for avgI, _, _, _, _,_ in data[-equilibrium_range:]])
            equilibrium_vals_bacteria.append(equilibrium_avg_bacteria)
            equilibrium_vals_immune.append(equilibrium_avg_immune)
            param_vals.append(val)


        ax1.plot(param_vals, equilibrium_vals_bacteria, label=f"Equilibrium {param}", marker='o')
        ax1.set_title(f"Equilibrium Analysis for {param} -Bacteria")
        ax1.set_xlabel(f"Value of {param}")
        ax1.set_ylabel("Average bacteria cell count at equilibrium")
        ax1.legend()
        ax2.plot(param_vals, equilibrium_vals_immune, label=f"Equilibrium {param}", marker='o', color='orange')
        ax2.set_title(f"Equilibrium Analysis for {param} -Immune Cells")
        ax2.set_xlabel(f"Value of {param}")
        ax2.set_ylabel("Average immune cell count at equilibrium")
        ax2.legend()
        plt.tight_layout()
        filename = f"equilibrium_interaction_{param}.png"
        plt.savefig(f"data/{filename}")
        plt.show()


def sensitivityTask(maxIter, temp_parms, runs, boxLen, immuneFrac):
    """
    Perform a sensitivity analysis task for the set of parameters

    Parameters:
    - maxIter (int):       Maximum number of iterations for each simulation run.
    - temp_parms (dict):    Dictionary containing model parameters for a single simulation run.
    - runs (int):           Number of simulation runs to perform.
    - boxLen (float):       Length of the simulation box.
    - immuneFrac (float):   Fraction of immune cells in the initial conditions.

    Returns
    List of simulation results for each run.

    """
    output = []
    temp_parms = {key: value for key, value in temp_parms.items() if key  not in ('length','width')}
    for _ in range(runs):
        output.append(runOnce(maxIter, temp_parms, boxLen, immuneFrac))
    return calculate_statistics(output)


def sensitivityAnalysis(model_class, param_ranges, initial_conditions, maxIter, runs, processes, boxLen, immuneFrac):
    """
    Perform sensitivity analysis for multiple parameters in parallel.

    Parameters:
    - model_class:                  Class of the model to be used for simulation.
    - param_ranges (dict):          Dictionary containing parameter ranges for sensitivity analysis.
    - initial_conditions (dict):    Dictionary containing initial conditions for the model.
    - maxIter (int):                Maximum number of iterations for each simulation run.
    - runs (int):                   Number of simulation runs to perform for each parameter value.
    - processes (int):              Number of parallel processes to use for simulation.
    - boxLen (float):               Length of the simulation box.
    - immuneFrac (float):           Fraction of immune cells in the initial conditions.

    Returns
    Dict containing sensitivity analysis results.
    """
    sensitivity_results = {}
    tasks = []
    for param, (min_val, max_val) in param_ranges.items():
        step = (max_val - min_val) / 10
        for val in np.arange(min_val, max_val + step, step):
            temp_parms = initial_conditions.copy()
            temp_parms['length'] = boxLen
            temp_parms['width'] = boxLen
            temp_parms[param] = val
            tasks.append((maxIter, temp_parms, runs, boxLen, immuneFrac))

    # to speed up, use multiple processes
    with Pool(processes) as pool:
        results = pool.starmap(sensitivityTask, tasks)

    i = 0
    for param, (min_val, max_val) in param_ranges.items():
        step = (max_val - min_val) / 10
        sensitivity_results[param] = []
        for val in np.arange(min_val, max_val + step, step):
            sensitivity_results[param].append((val, results[i]))
            i += 1

    return sensitivity_results



def parScan(maxIter:int, parms: Dict[str, List[any]], runs: int, processes: int, boxLen: int, immuneFrac: float,
            filename: str) -> Tuple[List[Dict[str, any]], List[List[Tuple[float, float, float, float, float, float]]]]:
    """
    Loop through possible parameter combinations and perform paralell runs of them.

    Args:
        maxIter          (int): The maximum amount of iterations per run.
        parms (Dict[str, any]): Dict of list for all kwargs to be scanned.
        runs             (int): The amountof runs per parameter combination to take average of.
        processes        (int): The amount of available processes to run in parallel.
        boxLen           (int): The length of the rectangular box for the spatial model.
        immuneFrac     (float): The amount of cells occupied by immune cells
        filename         (str): Filename/ relative filepath from data/ for result files.

    """

    for key, value in parms.items():
        if not isinstance(value, list):
            parms[key] = [value]
    parmsList: List[Dict[str,float]] = []
    outputs   = []
    for option0 in parms["pImmuneKill"]:
        for option1 in parms["pBacteriaMult"]:
            for option2 in parms["pBacteriaSpawn"]:
                parmsList.append({"pImmuneKill" : option0,
                                  "pBacteriaMult" : option1,
                                  "pBacteriaSpawn": option2
                                 })

                result = paralelRun(maxIter, parmsList[-1], runs, processes, boxLen, immuneFrac)
                outputs.append(result)
                saveResults(outputs[-1], filename, parmsList[-1], runs, maxIter)
    return parmsList, outputs


def paralelRun(maxIter:int, parms: Dict[str, float], runs: int, processes: int, boxLen: int,
               immuneFrac: float) -> List[Tuple[float, float, float, float, float, float]]:
    """
    Parralellized data collection function for BacteriaImmuneModel

    Measures the occupancy of immune cells as defined by nImmune / boxLength**2 per iteration for a
    set amount of runs. Average, variance and standard deviation are calculated from this and returned.

    Args:
        maxIter          (int): The maximum amount of iterations per run.
        parms (Dict[str, any]): Dict of kwargs for BacteriaImmuneModel.
        runs             (int): The amountof runs per parameter combination to take average of.
        processes        (int): The amount of available processes to run in parallel.
        boxLen           (int): The length of the rectangular box for the spatial model.
        immuneFrac     (float): The amount of cells occupied by immune cells
        filename         (str): Filename/ relative filepath from data/ for result files.
    """

    t_init = time()

    a = multiprocessing.Pool(processes)

    input = [(maxIter, parms, boxLen, immuneFrac) for i in range(runs)]

    print(f"Running {runs} simulations of {maxIter} iterations over {processes} processes")
    output = a.starmap(runOnce, input)
    print(f"Total model runtime: {time() - t_init}")
    t_init = time()
    output = calculate_statistics(output)
    print(f"Total result processing runtime: {time() - t_init}")

    return output


def runOnce(maxIter: int, parms: Dict[str, float], boxLen: int, immuneFrac: float) -> "np.ndarray[float]":
    """
    Runs the model once for the given parameter.

    Args:
        maxIter (int): The amount of iterations to run the model for.
        INTIAL_IMMUNE (float): Initial amount of immune cells.
        parms (Dict[str, float]): Model parameters (pBacteriaMult, pImmuneKill, pBacteriaSpawn).

    Returns (np.ndarray[float]): A 2 x maxIter array with immune cell data in the left collumn and bacterial
        cell data in the right.
    """
    parms = copy(parms)
    model = bacteriaImmuneModel(boxLen, boxLen, **parms)
    model.seedImmune(round(boxLen**2 * immuneFrac))
    vals: "np.ndarray[float]" = np.ndarray((2,maxIter), float)
    for iter in range(maxIter):
        model.timestep()
        vals[0, iter] = model.get_nImmuneCells() / boxLen**2
        vals[1, iter] = model.get_nbacteriaCells() / boxLen**2
    return vals


def calculate_statistics(values: "List[np.ndarray[float]]") -> List[Tuple[float, float, float, float, float, float]]:
    """
    Calculates statistical measures for a given list of values.

    Args:
        values (List[np.ndarray[float]): List of arrays containig the immunecell and bacterial cell
        data for every run of a pool.

    Returns:
        List[Tuple[float, float, float, float, float, float]]: A tuple containing the average, variance,
        and standard deviation of the immune cells [:3] and bacteria cells [3:] per iteration.
    """
    # Split List of Per run tuples of per cell tuples up into List of Per cell Lists of per run lists
    immune   = [point[0,:] for point in values]
    immune   = list(zip(*immune))
    bacteria = [point[1,:] for point in values]
    bacteria = list(zip(*bacteria))
    plots    = []

    for pointImm, pointBac in zip(immune, bacteria):
        averageImm  = sum(pointImm) / len(pointImm)
        varianceImm = sum((x - averageImm) ** 2 for x in pointImm) / len(pointImm)
        st_devImm   = sqrt(varianceImm)
        averageBac  = sum(pointBac) / len(pointBac)
        varianceBac = sum((x - averageBac) ** 2 for x in pointBac) / len(pointBac)
        st_devBac   = sqrt(varianceBac)

        plots.append((averageImm, varianceImm, st_devImm, averageBac, varianceBac, st_devBac))


    return plots


def multiPlot(results: List[Tuple[float, float, float, float, float, float]], plotTitle= None) -> None:
    """
    Plots the results of model simulations.

    Args:
        results: List[Tuple[float, float, float, float, float, float]]: A tuple containing the average,
        variance, and standard deviation of the immune cells [:3] and bacteria cells [3:] per iteration.
    """
    fig, ax = plt.subplots()
    assert isinstance(ax, plt.Axes)

    iters = range(len(results))
    averagesI = [avgI    for avgI, varI, stdDevI, avgB, varB, stdDevB in results]
    std_devsI = [stdDevI for avgI, varI, stdDevI, avgB, varB, stdDevB in results]
    averagesB = [avgB    for avgI, varI, stdDevI, avgB, varB, stdDevB in results]
    std_devsB = [stdDevB for avgI, varI, stdDevI, avgB, varB, stdDevB in results]

    for averages, std_devs, label in [(averagesI, std_devsI, "Immune Cells"), (averagesB, std_devsB, "Bacteria")]:
        ax.plot(iters, averages, label=f"Average {label}")
        ax.fill_between(iters,
                        [a - s for a, s in zip(averages, std_devs)],
                        [a + s for a, s in zip(averages, std_devs)],
                        alpha=0.5, label=f"Std Dev {label}"
                       )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cell occupancy")
    ax.legend()
    if plotTitle:
        ax.set_title(plotTitle)

    plt.tight_layout()
    plt.show()

def multiMultiplot(parms: List[Dict[str, float]], results: List[List[Tuple[float, float, float, float]]]):
    fig, axs = plt.subplots(1, len(results))

    for ax, result, parm, id in zip(axs, results, parms, range(len(results))):
        iters = range(len(result))
        averagesI = [avgI       for avgI, varI, stdDevI, avgB, varB, stdDevB in result]
        std_devsI = [stdDevI    for avgI, varI, stdDevI, avgB, varB, stdDevB in result]
        averagesB = [avgB       for avgI, varI, stdDevI, avgB, varB, stdDevB in result]
        std_devsB = [stdDevB    for avgI, varI, stdDevI, avgB, varB, stdDevB in result]

        for averages, std_devs, label in [(averagesI, std_devsI, "Immune Cells"), (averagesB, std_devsB, "Bacteria")]:
            assert isinstance(ax, plt.Axes)
            ax.plot(iters, averages, label=f"Average {label}")
            ax.fill_between(iters,
                            [a - s for a, s in zip(averages, std_devs)],
                            [a + s for a, s in zip(averages, std_devs)],
                            alpha=0.5, label=f"Std Dev {label}"
                        )
            ax.set_title(f"Run {id}\n{str(parm.items())[10:]}", fontsize=10)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cell occupancy")
        ax.legend()

    plt.tight_layout()
    plt.show()

def plotPhaseSpace(results, param_values, param_name, cmap='viridis'):
    """
    Plots a phase space diagram of immune cells vs. bacteria cells with color gradients based on a parameter value.

    Args:
        results (List[Dict]): List of result dictionaries, each containing 'immune' and 'bacteria' lists.
        param_values (float or List[float]): Single parameter value or list of parameter values corresponding to each result set in `results`.
        param_name (str): Name of the parameter that varies between simulations.
        cmap (str): Name of the matplotlib colormap to use for the gradient.
    """
    if not isinstance(param_values, (list, np.ndarray)):
        param_values = [param_values]
    
    norm = mcolors.Normalize(vmin=min(param_values), vmax=max(param_values))
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)

    plt.figure(figsize=(10, 6))
    
    for result, param_value in zip(results, param_values):
        immune_averages = [avgI for avgI, _, _, _, _, _ in result]
        bacteria_averages = [avgB for _, _, _, avgB, _, _ in result]
        color = scalar_map.to_rgba(param_value)
        
        plt.plot(immune_averages, bacteria_averages, '-o', markersize=5, lw=2, color=color, 
                 label=f'{param_name}: {param_value}')
    
    plt.title("Phase Plot of Immune Cells vs. Bacteria Cells")
    plt.xlabel("Average Immune Cell Fraction")
    plt.ylabel("Average Bacteria Cell Fraction")
    plt.colorbar(scalar_map, label=param_name)
    plt.legend()
    plt.grid(True)
    plt.show()


def saveResults(plot: List[Tuple[float, float, float]], filename: str, parms: Dict[str, float],
                runs: int, maxIter: int
               ) -> str:
    """
    Saves the calculated results to a CSV file.

    Args:
        plot (List[Tuple[float, float, float]]): A list of tuples containing the average occupation,
            the variance and the standard deviation for every iteration.
    """
    os.makedirs("data", exist_ok=True)

    filepath = f"data/{filename+  f'-{str(parms.items())[10:]}'.replace(' ', '')}.csv"

    with open(filepath, "w") as outFile:

        outFile.write(f"{parms}\n")
        outFile.write(f"{runs} runs over {maxIter} iterations\n")
        outFile.write("\n")
        outFile.write("AverageImmune, VarianceImmune, StdDefImmune, AverageBacteria, VarianceBacteria, StdDefBacteria\n")
        for avgI, varI, stdDefI, avgB, varB, stdDefB in plot:
            outFile.write(f"{avgI},{varI},{stdDefI},{avgB},{varB},{stdDefB}\n")
        outFile.write("\n")
    return filepath

def readData(filename: str) -> List[Tuple[float, float, float, float, float]]:
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

def main(args: List[str]):
    """
    Main function, see file docstring

    Takes a list of arguments as input.
    """
    mode, maxIter, parms, runs, processes, boxLen, immuneFrac, filename, plotTitle, noPlot = parseArgv(args)

    if mode == "sensitivity":
        param_ranges = {
            'pImmuneKill': (0.1, 1.0),
            'pbacteriaMult': (0.1, 0.7),
            'pbacteriaSpawn': (0.05, 0.4)
        }
        initial_conditions = {'length': boxLen, 'width': boxLen,
                              'pImmuneKill': parms.get('pImmuneKill', 0.5),
                              'pbacteriaMult': parms.get('pbacteriaMult', 0.4),
                              'pbacteriaSpawn': parms.get('pbacteriaSpawn', 0.1)}
        sensitivity_results = sensitivityAnalysis(bacteriaImmuneModel, param_ranges, initial_conditions, maxIter,
                                                  runs, processes, boxLen, immuneFrac)
        plotEquilibriumAnalysis(sensitivity_results, equilibrium_range=50)
    elif mode == PLOT:
        multiPlot(readData(filename), plotTitle)
        return
    elif mode == PARAMETER_SCAN:
        output = parScan(maxIter, parms, runs, processes, boxLen, immuneFrac, filename)
        if not noPlot:
            multiMultiplot(*output)
        return
    output = paralelRun(maxIter, parms, runs, processes, boxLen, immuneFrac)
    saveResults(output, filename, parms, runs, maxIter)
    if not noPlot:
        multiPlot(output)
    return output

def parseArgv(args: List[str]):
        """
        Argv parser.

        returns mode, maxIter, parms, runs, processes, boxLen, immuneFrac, filename, plotTitle, noPlot
        """
        args = copy(args)
        mode       = RUN_SINGLE_PARM
        runs       = RUNS
        maxIter    = MAX_ITER
        filename   = DEF_FILENAME
        processes  = PROCESSES
        boxLen     = LEN
        immuneFrac = INITIAL_IMMUNE
        plotTitle  = DEF_FILENAME
        parms      = PARMS
        noPlot     = False


        if len(args) > 1:
            # Plotting past results:
            if args[1].lower() in ['p', 'plot']:
                mode = PLOT
                filename = args[2]

                if len(args) > 3:
                    plotTitle = args[3]

            # Run parallel program
            else:
                args.pop(0)
                if args[0].isnumeric():
                    runs      = int(args.pop(0))
                if len(args) > 0 and args[0].isnumeric():
                    maxIter   = int(args.pop(0))
                if len(args) > 0 and args[0].isnumeric():
                    processes = int(args.pop(0))
                if len(args) > 0 and args[0] == 'sensitivity':
                    mode = "sensitivity"
                    return mode, maxIter, parms, runs, processes, boxLen, immuneFrac, filename, plotTitle, noPlot

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
                elif key.lower() in ["noplot", "noplotting"]:
                    noPlot = True

                else:
                    if "[" in val:
                        parms[key] = json.loads(val)
                        mode = PARAMETER_SCAN
                    else:
                        if "[" in val:
                            parms[key] = json.loads(val)
                            mode = PARAMETER_SCAN
                        else:
                            parms[key] = float(val)
        return mode, maxIter, parms, runs, processes, boxLen, immuneFrac, filename, plotTitle, noPlot

if __name__ == "__main__":
    from sys import argv

    main(argv)
