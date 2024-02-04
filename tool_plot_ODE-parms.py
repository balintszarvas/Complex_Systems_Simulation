import matplotlib.pyplot as plt
import glob
from math import exp

from typing import List

""" 
Quick plotting program for ODE_batch output files, for variable multiplication probability only.
"""

fig, axes = plt.subplots(2,2) 
XLABEL = "Bacteria Multiplication Probability"

graphs = [[], [], [], [], []]
for filename in glob.glob("data\Batch_ODE-fitted_runs\ODEPARMS_*.csv"):
    with open(filename) as parmFile:
        line = parmFile.readline()
        while line:
            print(line)
            splitLine = line.split(',')
            splitLine = [float(val) for val in splitLine]
            
            for graph, item in zip(graphs, splitLine):
                graph.append(item)

            line = parmFile.readline()

# ALPHA
mult = graphs[0]
alphas = graphs[1]
ax = axes[0,0]

alphaline = [-3.0738 * x + 0.6586 for x in mult] # Trendline obtained from Excel

assert(isinstance(ax, plt.Axes))
ax.scatter(mult, alphas)
ax.plot(mult, alphaline, "--")
ax.set_ybound(0, 0.8)
ax.set_xlabel(XLABEL)
ax.set_ylabel("Alpha")
ax.set_title("Alpha/ Immune Kill Rate")

# BETA
ax = axes[0,1]
betas = graphs[2]
avgBeta = sum([beta for beta in betas if beta > 0]) / len([beta for beta in betas if beta > 0])
print(avgBeta)

assert(isinstance(ax, plt.Axes))
ax.scatter(mult, betas, color="tab:orange")
ax.axhline(avgBeta, 0, 1, linestyle="--", color="tab:orange")
ax.set_xlabel(XLABEL)
ax.set_ylabel("Beta")
ax.set_title("Beta/ Immune overcrowding factor")
ax.set_ybound(0, 0.16)

# GAMMA
ax = axes[1,0]
gammas = graphs[3]
gammaline = [-17.519 * x + 1.1353 for x in mult] # Trendline obtained from Excel

assert(isinstance(ax, plt.Axes))
ax.scatter(mult, gammas, color="tab:red")
ax.plot(mult, gammaline, "--", color="tab:red")
ax.axhline(0, color="k", linewidth=.5)
ax.set_xlabel(XLABEL)
ax.set_ylabel("Gamma")
ax.set_title("Gamma/ Bacteria reproduction rate")
ax.set_ybound(-1.4, 1.6)

# SPAWN
ax = axes[1,1]
spawns = graphs[4]
spawnLine = [1e-6 * exp(42.364 * x) for x in mult] # Trendline obtained from Excel

assert(isinstance(ax, plt.Axes))
ax.scatter(mult, spawns, color="tab:green")
ax.plot(mult, spawnLine, "--", color="tab:green")
ax.axhline(0, color="k", linewidth=.5)
ax.set_xlabel(XLABEL)
ax.set_ylabel("Spawn")
ax.set_title("Spawn/ Bacteria introduction rate")
ax.set_ybound(0, 1e-4)

print("Done")
plt.show()



