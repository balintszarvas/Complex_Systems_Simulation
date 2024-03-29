import tool_paralelRun as par
import tool_fit_ODE as fit
import numpy as np

"""Simple program that generates desired model plots and fits them to an ODE

Takes variable pBacteriaMult, and fixed ImmuneKill and pBacteriaSpawn parameters
Places files in ./data/Batch_ODE-fitted_runs/
Creates files for individual runs and a file containing all ODE parameters as function of pBacteriaMult 
"""

SAMPLES = 20
MIN     = 0.01
MAX     = 0.15
GUESS   = np.array([1.5, 0.09315, 0.71, 2.875e-6, 0.1])
OUTDIR  =  "Batch_ODE-fitted_runs/"
FILENAME = "EXP"

MAXITER   = 2000
PROCESSES = 8
RUNS      = 8
LEN       = 200
DT        = 0.1


def main():
    """
    Takes variable pBacteriaMult, and fixed ImmuneKill and pBacteriaSpawn parameters
    Places files in ./data/Batch_ODE-fitted_runs/
    Creates files for individual runs and a file containing all ODE parameters as function of pBacteriaMult 
    """
    pBacteriaMult  = [MIN + i * ((MAX - MIN) / SAMPLES) for i in range(SAMPLES)]
    pImmuneKill  = 1.0
    pBacteriaSpawn = 0.01

    filenames = []
    for mult in pBacteriaMult:
        parms = {"pBacteriaMult":mult, "pImmuneKill": pImmuneKill, "pBacteriaSpawn": pBacteriaSpawn}

        dataOut = par.paralelRun(MAXITER, parms, RUNS, PROCESSES, LEN, 0.006)
        filenames.append(par.saveResults(dataOut, OUTDIR + FILENAME, parms, RUNS, MAXITER))

    xOuts = []

    for filename in filenames:
        xOut = fit.fitODE(*fit.loadData(filename), GUESS, DT)["x"]
        print(f"{filename} DONE")
        xOuts.append(xOut)

    with open("data/" + OUTDIR + "ODEPARMS_" + FILENAME, "w") as outFile:
        for mult, xOut in zip(pBacteriaMult, xOuts):
            outFile.write(f"{mult}, {str([item for item in xOut])[1:-1]}\n")
    return

if __name__ == "__main__":
    main()