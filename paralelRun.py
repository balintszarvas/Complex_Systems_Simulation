from copy import copy
from math import sqrt
from time import time, sleep
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
        
        case sensitive model parameters:
            pImmuneKill=(float/list[float])
            - The chance an immune cell kills a cancer cell it occupies the same cell as.

            pCancerMultiply=(float/list[float])
            - The chance a cancer cell multiplies.

            pCancerSpawn=(float/list[float])
            - The chance a cancer cell spawns on the grid.

            A list structure can also be entered, which makes the program loop through all possible
            combinations of these parameters.
"""


LEN           = 200  # Box lenght
MAX_ITER      = 10 # Amount of iterations per model run
PROCESSES     = 4    # Amount of paralel processes able to run the model
RUNS          = 4    # Amount of runs to be done
DEF_FILENAME  = "TEMP"

INITIAL_IMMUNE = 0.006 # Initial amount of immune cells (arbitrary in theory)

# Model parameters
PARMS = { # Maybe rework into generator function
        "pImmuneKill" : 0.5,
        "pCancerMult" : 0.4,
        "pCancerSpawn": 0.1
        }


def parScan(maxIter:int, parms: Dict[str, any], runs: int, processes: int, boxLen: int, immuneFrac: float, filename: str):
    """Loop through possible parameters"""

    for key, value in parms.items():
        if not isinstance(value, list):
            parms[key] = [value]
    parmsList: List[Dict[str,float]] = []
    outputs   = []
    for option0 in parms["pImmuneKill"]:
        for option1 in parms["pCancerMult"]:
            for option2 in parms["pCancerSpawn"]:
                parmsList.append({"pImmuneKill" : option0,
                                  "pCancerMult" : option1,
                                  "pCancerSpawn": option2
                                 })
                
                result = paralelRun(maxIter, parmsList[-1], runs, processes, boxLen, immuneFrac)
                outputs.append(result)
                newName = filename + f"-{str(parmsList[-1].items())[10:]}"
                saveResults(outputs[-1], newName, parmsList[-1], runs, maxIter)
    # TODO add multiple side by side plots
    return parmsList, outputs


def paralelRun(maxIter:int, parms: Dict[str, float], runs: int, processes: int, boxLen: int, immuneFrac: float) -> None:
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

    print(f"Running {runs} simulations of {maxIter} iterations over {processes} processes")
    output = a.starmap(runOnce, input)    
    print(f"Total model runtime: {time() - t_init}")
    output = calculate_statistics(output)
    print(f"Total result processing runtime: {time() - t_init}")

    return output


def runOnce(maxIter: int, parms: Dict[str, float], boxLen: int, immuneFrac: float) -> List[Tuple[float, float]]:
    """
    Runs the model once for the given parameter.
    
    Args:
        maxIter (int): The amount of iterations to run the model for.
        INTIAL_IMMUNE (float): Initial amount of immune cells.
        parms (Dict[str, float]): Model parameters (pCancerMult, pImmuneKill, pCancerSpawn).

    Returns (List[Tuple[float]]): A list of tuples containing a list of immune and bacteria occupancy values.
    """
    parms = copy(parms)
    model = CancerImmuneModel(boxLen, boxLen, **parms)
    model.seedImmune(round(boxLen**2 * immuneFrac))
    vals: List[Tuple[float, float]] = []
    for iter in range(maxIter):
        model.timestep()
        point = (model.get_nImmuneCells() / boxLen**2, model.get_nCancerCells() / boxLen**2)
        vals.append(point)
    return vals


def calculate_statistics(values: List[Tuple[float, float]]) -> Tuple[float, float, float, float, float, float]:
    """
    Calculates statistical measures for a given list of values.

    Args:
        values (List[float]): List of numerical values.

    Returns:
        Tuple[float, float, float]: A tuple containing the average, variance, and standard deviation 
        of the input values.
    """
    # Split List of Per run tuples of per cell tuples up into List of Per cell Lists of per run lists 
    immune   = [[item[0] for item in point] for point in values]
    bacteria = [[item[1] for item in point] for point in values]
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


def multiPlot(results: List[Tuple[float, float, float, float]], plotTitle= None) -> None:
    """
    Plots the results of model simulations.

    Args:
        plot (List[Dict[str, List[Tuple[float, float, float]]]]): The calculated statistics for immune 
        and cancer cell occupancy.

    Returns:
        None: This function does not return anything but plots the results.
    """
    fig, ax = plt.subplots()

    iters = range(len(results))
    averagesI = [avgI    for avgI, varI, stdDevI, avgB, varB, stdDevB in results]
    std_devsI = [stdDevI for avgI, varI, stdDevI, avgB, varB, stdDevB in results]
    averagesB = [avgB    for avgI, varI, stdDevI, avgB, varB, stdDevB in results]
    std_devsB = [stdDevB    for avgI, varI, stdDevI, avgB, varB, stdDevB in results]

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


def saveResults(plot: List[Tuple[float, float, float]], filename: str, parms: Dict[str, float], 
                runs: int, maxIter: int
               ) -> None:
    """
    Saves the calculated results to a CSV file.

    Args:
        plot (List[Tuple[float, float, float]]): A list of tuples containing the average occupation,
            the variance and the standard deviation for every iteration.
    """
    os.makedirs("Output", exist_ok=True)

    with open(f"output/{filename}.csv", "w") as outFile:
          
        outFile.write(f"{parms}\n")
        outFile.write(f"{runs} runs over {maxIter} iterations\n")
        outFile.write("\n")
        outFile.write("AverageImmune, VarianceImmune, StdDefImmune, AverageBacteria, VarianceBacteria, StdDefBacteria\n")
        for avgI, varI, stdDefI, avgB, varB, stdDefB in plot:
            outFile.write(f"{avgI},{varI},{stdDefI},{avgB},{varB},{stdDefB}\n")
        outFile.write("\n")
    return

if __name__ == "__main__":
    from sys import argv
    import json

    RUN_SINGLE_PARM = 0
    PLOT = 1
    PARAMETER_SCAN = 2

    def main(args: List[str]):
        """
        Main function, see file docstring
        
        Takes argv as input.
        """
        mode, maxIter, parms, runs, processes, boxLen, immuneFrac, filename, plotTitle, noPlot = parseArgv(args)
        if mode == PLOT:
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
                    elif key.lower() in ["noplot"]:
                        noPlot = True
                    
                    else:
                        if "[" in val:
                            parms[key] = json.loads(val)
                            mode = PARAMETER_SCAN
                        else:
                            parms[key] = float(val)

        return mode, maxIter, parms, runs, processes, boxLen, immuneFrac, filename, plotTitle, noPlot

    main(argv)
