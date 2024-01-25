from .cancerImmuneModel import CancerImmuneModel
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from typing import List

class Visualizer:
    def __init__(self, model: CancerImmuneModel, threshold) -> None:
        self.threshold = threshold
        self.model = model
        self.latticeCells = (self.model.dim[0] * self.model.dim[1])
        fig, axs = plt.subplots(1, 2)
        ax0 = axs[0]
        ax1 = axs[1]
        # ax2 = axs[2]
        assert isinstance(ax0, plt.Axes)
        assert isinstance(ax1, plt.Axes)
        # assert isinstance(ax2, plt.Axes)

        self.fig = fig
        self.ax0 = ax0
        self.ax1 = ax1
        # self.ax2 = ax2
        self.fig.show()

        self.immuneCells: List[int] = []
        self.cancerCells: List[int] = []
        self.ind = 'no found'

    def frame(self, i):
            self.model.timestep()
            self.ax0.clear()
            self.ax0.imshow(self.model.cancerLattice * np.invert(self.model.immuneLattice.astype(bool)) + self.model.immuneLattice)

            self.immuneCells.append(self.model.get_nImmuneCells() / self.latticeCells)
            self.cancerCells.append(self.model.get_nCancerCells() / self.latticeCells)
            self.ax1.clear()
            self.ax1.plot(self.immuneCells, label="Immune")
            self.ax1.plot(self.cancerCells, label="Cancer")
            self.ax1.legend()
            # self.ax1.set_yscale("log")
            # self.ax1.set_xscale("log")

    def get_equilibrium_2(self):
        avg_list = []
        for ind in range(0, len(self.immuneCells)-10, 10):
            avg = 0
            for i in range(10):
                avg += self.immuneCells[ind]
            avg_list.append(avg)

        for ind in range(len(avg_list)):
            current_value = avg_list[ind]
            for j in range(i + 1, len(avg_list)):
                if avg_list[j] < current_value:
                    self.ind = ind
                    return 1
        return 2

    def get_no_cancer(self):
        for ind, value in enumerate(self.cancerCells):
            if self.cancerCells[ind] == 0 and self.cancerCells[ind+1] == 0:
                return ind
        return 2
    
    def get_equilibrium(self):
        start = self.get_no_cancer()
                
        for ind in range(start, len(self.cancerCells)-10):
            print(ind)
            if abs(self.cancerCells[ind] - self.cancerCells[ind+10]) < self.threshold:
                self.ind = ind
                return 1
        return 2  

    def run(self):
        ani = FuncAnimation(self.fig, self.frame, frames=None, interval=1, repeat = False)
        plt.show()

        