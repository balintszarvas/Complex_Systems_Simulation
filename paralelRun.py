from copy import copy
from math import sqrt
from time import time
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import multiprocessing

from model.classes.cancerImmuneModel import CancerImmuneModel

from typing import Dict, List, Tuple

# TODO: Rework global constants into parameter s
LEN           = 200  # Box lenght
MAX_ITER      = 800 # Amount of iterations per model run
PROCESSES     = 8    # Amount of paralel processes able to run the model
RUNS          = 15  # Amount of runs to be done

INTIAL_IMMUNECELL = [0.5] # Initial amount of immune cells (arbitrary in theory)
CANCER_NUMBER = [1] # Initial amount of cancer cells (arbitrary in theory)

# Model parameters
PARMS = { # Maybe rework into generator function
        "pImmuneKill" : [0.5],
        "pCancerMult" : [0.5],#[0, 0.01, 0.05, 0.075, 0.1, 0.2,  0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.575,  0.6, 0.625, 0.65, 0.675, 0.7, 0.75, 0.8, 0.9, 0.95, 1],
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
    params = []

    for cancermulti in PARMS["pCancerMult"]:
        for immunekill in PARMS["pImmuneKill"]:
            for cancerspawn in PARMS["pCancerSpawn"]:
                for initial_imm in INTIAL_IMMUNECELL:
                    current_parms = {
                        "pCancerMult": cancermulti,
                        "pImmuneKill": immunekill,
                        "pCancerSpawn": cancerspawn,
                        "InitialImmune": (initial_imm * LEN**2)
                    }
                    params.append(current_parms)
                    input = [(MAX_ITER, current_parms) for i in range(RUNS)]
                    output = a.starmap(runOnce, input)
                    output = avgResults(output)
                    results.append(output)
                    print(f"Finished run with parameters: {current_parms}")

    
    print(f"Total simulation runtime: {time() - t_init}")
    on_same_plot = True # Change to False if you want separate subplots
    multiPlot(results, params)
    #phasePlot(results, params, on_same_plot)
    #plotCancerFractionVsMulti(results, params)


def runOnce(maxIter: int, parms: Dict[str, float]) -> Dict[str, List[float]]:
    """
    Runs the model once for the given parameter.
    
    Args:
        maxIter (int): The amount of iterations to run the model for.
        parms (Dict[str, float]): Model parameters (pCancerMult, pImmuneKill, pCancerSpawn, InitialImmune).

    Returns:
        Dict[str, List[float]]: A dictionary with keys "Immune" and "Cancer", each containing a list of cell occupancy values.
    """
    parms = copy(parms)
    initial_immune = parms.pop('InitialImmune', 0)  # Extract initial immune cells, use 0 if not provided
    model = CancerImmuneModel(LEN, LEN, **parms)
    model.seedImmune(round(initial_immune))
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


def multiPlot(results: List[Dict[str, List[Tuple[float, float, float]]]], params: List[Dict[str, float]]) -> None:
    """
    Plots the results of model simulations with specific colors for cancer and immune cells.

    Args:
        results (List[Dict[str, List[Tuple[float, float, float]]]]): The calculated statistics for immune and cancer cell occupancy.
        params (List[Dict[str, float]]): The parameters used for each run.

    Returns:
        None: This function does not return anything but plots the results.
    """
    fig, axs = plt.subplots(1, len(results))

    axs = np.atleast_1d(axs)  # Make sure axs is a list of axes

    middle_subplot_index = len(results) // 2

    for i, result in enumerate(results):
        # Plotting for each cell type
        for cell_type in result:
            iters = range(len(result[cell_type]))
            averages = [avg for avg, _, _ in result[cell_type]]
            std_devs = [std_dev for _, _, std_dev in result[cell_type]]
            color = 'red' if cell_type == 'Cancer' else 'blue'
            cell_name = 'Bacteria' if cell_type == 'Cancer' else 'Immune'
            axs[i].plot(iters, averages, label=f"{cell_name} Average", color=color)
            axs[i].fill_between(iters, 
                                [a - s for a, s in zip(averages, std_devs)],
                                [a + s for a, s in zip(averages, std_devs)],
                                alpha=0.5, color=color, label=f"{cell_name} Std Dev")

        axs[i].set_title(f"{params[i]['pCancerMult']}")
        axs[i].set_ylim(0, 1)
        axs[i].set_xlim(0, MAX_ITER)
        axs[i].grid()

        axs[i].tick_params(axis='both', which='major', labelsize=10)

        # Hide y-axis labels for all but the first subplot
        if i > 0:
            axs[i].set_yticklabels([])

        # Hide x-axis labels for all but the first, middle, and last subplots
        if i not in [0, middle_subplot_index, len(results) - 1]:
            axs[i].set_xticklabels([])

    
    axs[middle_subplot_index].set_xlabel("Iteration")
    axs[0].set_ylabel("Cell occupancy")
    axs[0].legend()
    plt.show()


def phasePlot(results: List[Dict[str, List[Tuple[float, float, float]]]], params: List[Dict[str, float]], on_same_plot: bool) -> None:
    cmap = cm.viridis.reversed()
    norm = mcolors.Normalize(vmin=min(param['pCancerMult'] for param in params), vmax=max(param['pCancerMult'] for param in params))
    
    if on_same_plot:
        fig, ax = plt.subplots()
        for i, result in enumerate(results):
            cancer_averages = [avg for avg, _, _ in result["Cancer"]]
            immune_averages = [avg for avg, _, _ in result["Immune"]]
            parameters = params[i]
            color = cmap(norm(parameters['pCancerMult']))
            points = np.array([cancer_averages, immune_averages]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(np.array(cancer_averages))
            lc.set_linewidth(2)
            line = ax.add_collection(lc)
            ax.plot(cancer_averages[-1], immune_averages[-1], 'o', color="black", markersize=10)
            for start, end in zip(points[:-1:5], points[1::5]):
                ax.annotate('', xy=end[0], xytext=start[0],
                            arrowprops=dict(facecolor=color, edgecolor=color, arrowstyle="->"))

        ax.set_xlabel("Bacterial Cell Fraction")
        ax.set_xlim(0, 1)
        ax.set_ylabel("Immune Cell Fraction")
        ax.set_ylim(0, 1)
        ax.set_title("Bacterial vs Immune Cell Fractions")
        ax.grid()
    else:
        fig, axs = plt.subplots(1, len(results), figsize=(15, 5))
        for i, result in enumerate(results):
            cancer_averages = [avg for avg, _, _ in result["Cancer"]]
            immune_averages = [avg for avg, _, _ in result["Immune"]]
            parameters = params[i]
            color = cmap(norm(parameters['pCancerMult']))
            points = np.array([cancer_averages, immune_averages]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(np.array(cancer_averages))
            lc.set_linewidth(2)
            axs[i].add_collection(lc)
            for start, end in zip(points[:-1:5], points[1::5]):
                axs[i].annotate('', xy=end[0], xytext=start[0],
                                arrowprops=dict(facecolor=color, edgecolor=color, arrowstyle="->"))
            axs[i].set_xlim(0, 1)
            axs[i].set_ylim(0, 1)
            axs[i].set_title(f"Run {i+1} Parameters: {parameters}")
            if i == len(results) // 2:  # Only set label for middle subplot
                axs[i].set_xlabel("Cancer Cell Fraction")
            if i == 0:
                axs[i].set_ylabel("Immune Cell Fraction")
            axs[i].legend()

        plt.tight_layout()

    plt.show()

def plotCancerFractionVsMulti(results: List[Dict[str, List[Tuple[float, float, float]]]], params: List[Dict[str, float]]) -> None:
    # Extracting final cancer fractions and corresponding multiplication rates
    cancer_multi_rates = []
    final_cancer_fractions = []
    
    for i, result in enumerate(results):
        final_cancer_fraction = result['Cancer'][-1][0] # Last average cancer fraction
        cancer_multi_rate = params[i]['pCancerMult']
        
        cancer_multi_rates.append(cancer_multi_rate)
        final_cancer_fractions.append(final_cancer_fraction)
    
    # Plotting
    fig, ax = plt.subplots()
    ax.plot(cancer_multi_rates, final_cancer_fractions, 'o-')
    ax.set_xlabel('Bacteria Multiplication Rate')
    ax.set_xlim(0, 1)
    ax.set_ylabel('Final Bacteria Fraction')
    ax.set_ylim(0, 1)
    ax.set_title('Final Bacteria Fraction vs Bacteria Multiplication Rate')
    ax.grid()
    plt.show()



def saveResults(results: List[Dict[str, List[Tuple[float, float, float]]]], 
                params: List[Dict[str, float]]) -> None:
    """
    Saves the calculated results and their corresponding parameters to a CSV file.

    Args:
        results (List[Dict[str, List[Tuple[float, float, float]]]]): The calculated statistics for immune and cancer cell occupancy.
        params (List[Dict[str, float]]): The parameters used for each run.

    Returns:
        None: This function does not return anything but saves the results to a file.
    """
    os.makedirs("Output", exist_ok=True)

    with open("output/results.csv", "w") as outFile:
        header = "pCancerMult,pImmuneKill,pCancerSpawn,InitialImmune,AvgImmune,VarImmune,StdDevImmune,AvgCancer,VarCancer,StdDevCancer\n"
        outFile.write(header)

        for i, param in enumerate(params):
            line = f"{param['pCancerMult']},{param['pImmuneKill']},{param['pCancerSpawn']},{param['InitialImmune']}"
            for cell_type in results[i]:
                avg, var, std_dev = results[i][cell_type][-1]  # Taking the last iteration values
                line += f",{avg},{var},{std_dev}"
            outFile.write(line + "\n")


if __name__ == "__main__":
    main()