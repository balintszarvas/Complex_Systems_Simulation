from copy import copy
from math import sqrt
from time import time
import os

import multiprocessing
import matplotlib.pyplot as plt

from model.classes.cancerImmuneModel import CancerImmuneModel

from typing import Dict, List, Tuple

# TODO: Rework global constants into parameters
LEN           = 100  # Box lenght
MAX_ITER      = 200 # Amount of iterations per model run
PROCESSES     = 8    # Amount of paralel processes able to run the model
RUNS          = 10    # Amount of runs to be done

INTIAL_IMMUNECELL = [0.1] # Initial amount of immune cells (arbitrary in theory)

# Model parameters
PARMS = { # Maybe rework into generator function
        "pImmuneKill" : [0.5],
        "pCancerMult" : [0.4, 0.5, 0.6],
        "pCancerSpawn": [0.1],
        }

def main() -> None:
    """
    Parralellized data collection program for CancerImmuneModel

    Measures the occupancy of immune cells as defined by nImmune / boxLength**2 per iteration for a
    set amount of runs. Average, variance and standard deviation are calculated from this and saved
    to ./output/temp.CSV.

    Average and standard deviation are plotted in matplotlib and displayed to the user.

    TODO:
        - Rework global constants into input parameters
        - Dynamic file naming
        - Parameter generator function for grid scanning
        - Input arguments (eg. to disable matplotlib figure display for shell automation)
    """
    
    t_init = time()
    a = multiprocessing.Pool(PROCESSES)

    results = []

    for cancermulti in PARMS["pCancerMult"]:
        for immunekill in PARMS["pImmuneKill"]:
            for cancerspawn in PARMS["pCancerSpawn"]:
                for initial_imm in INTIAL_IMMUNECELL:
                    current_parms = {
                        "pCancerMult": cancermulti,
                        "pImmuneKill": immunekill,
                        "pCancerSpawn": cancerspawn
                    }
                    INTIAL_IMM = initial_imm * LEN**2

                    input = [(MAX_ITER, INTIAL_IMM, current_parms) for i in range(RUNS)]
                    output = a.starmap(runOnce, input)
                    output = avgResults(output)
                    results.append(output)

    multiPlot(results)
    phasePlot(results)
    print(f"Total result processing runtime: {time() - t_init}")


def runOnce(maxIter: int, INTIAL_IMMUNE, parms: Dict[str, float]) -> Dict[float, float]:
    """
    Runs the model once for the given parameter.
    
    Args:
        maxIter           (int): The amount of iterations to run the model for
        parms (Dict[str,float]): Key-value pairs for the initializer described in .model.cancerImmuneModel
                                 barring length and with.
    
    Returns (Dict[float]): Immune cell and cancer cell occupancy relative to the surface of the lattice
    """
    parms = copy(parms)
    model = CancerImmuneModel(LEN, LEN, **parms)
    model.seedImmune(round(INTIAL_IMMUNE))
    vals = {"Immune": [], "Cancer": []}
    for iter in range(maxIter):
        model.timestep()
        vals["Immune"].append(model.get_nImmuneCells() / LEN**2)
        vals["Cancer"].append(model.get_nCancerCells() / LEN**2)

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
    fig, axs = plt.subplots(1, len(results))

    for i, result in enumerate(results):
        for cell_type in result:
            iters = range(len(result[cell_type]))
            averages = [avg for avg, _, _ in result[cell_type]]
            std_devs = [std_dev for _, _, std_dev in result[cell_type]]

            axs[i].plot(iters, averages, label=f"{cell_type} Average")
            axs[i].fill_between(iters, 
                                [a - s for a, s in zip(averages, std_devs)],
                                [a + s for a, s in zip(averages, std_devs)],
                                alpha=0.5, label=f"{cell_type} Std Dev")

        axs[i].set_ylim(0, 1)
        axs[i].set_title(f"Run {i+1}")
        axs[i].set_xlabel("Iteration")
        axs[i].set_ylabel("Cell occupancy")
        axs[i].legend()

    plt.tight_layout()
    plt.show()

def phasePlot(results: List[Dict[str, List[Tuple[float, float, float]]]]) -> None:
    """
    Plots the cell fractions against each other with initial conditions to show phase space and find fixed points.
    The x axis is the cancer cell fraction and the y axis is the immune cell fraction.
    """
    fig, axs = plt.subplots(1, len(results))

    for i, result in enumerate(results):
        cancer_averages = [avg for avg, _, _ in result["Cancer"]]
        immune_averages = [avg for avg, _, _ in result["Immune"]]

        axs[i].set_ylim(0, 1)
        axs[i].set_xlim(0, 1)
        axs[i].plot(cancer_averages, immune_averages, label="Cancer vs Immune")
        axs[i].set_xlabel("Cancer Cell Fraction")
        axs[i].set_ylabel("Immune Cell Fraction")
        axs[i].set_title(f"Run {i+1}")
        axs[i].legend()

    plt.tight_layout()
    plt.show()


def saveResults(plot: Dict[str, List[Tuple[float, float, float]]]) -> None:
    """
    Saves the calculated results to a CSV file.

    Args:
        plot (Dict[str, List[Tuple[float, float, float]]]): The calculated statistics for immune and cancer cell occupancy.

    Returns:
        None: This function does not return anything but saves the results to a file.
    """
    os.makedirs("Output", exist_ok=True)

    with open("output/TEMP.csv", "w") as outFile:
          
        outFile.write(f"{PARMS}\n")
        outFile.write(f"{RUNS} runs over {MAX_ITER} iterations\n")
        outFile.write("\n")
        outFile.write("AverageImmune, VarianceImmune, StdDefImmune, AverageCancer, VarianceCancer, StdDefCancer\n")
        for CellType in plot.keys():
            for (avg, var, stdDef) in plot[CellType]:
                outFile.write(f"{avg}, {var}, {stdDef}")
            outFile.write('\n')
        outFile.write("\n")
    return


if __name__ == "__main__":
    main()