from .cancerImmuneModel import CancerImmuneModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from typing import List

class Visualizer:
    def __init__(self, model: CancerImmuneModel) -> None:
        self.model = model
        fig, axs = plt.subplots(1, 2)
        ax0 = axs[0]
        ax1 = axs[1]
        assert isinstance(ax0, plt.Axes)
        assert isinstance(ax1, plt.Axes)

        self.fig = fig
        self.ax0 = ax0
        self.ax1 = ax1
        self.fig.show()

        self.immuneCells: List[int] = []
        self.cancerCells: List[int] = []
    
    def frame(self, i):
            self.model.timestep()
            self.ax0.clear()
            self.ax0.imshow(self.model.cancerLattice * np.invert(self.model.immuneLattice.astype(bool)) + self.model.immuneLattice) 

            self.immuneCells.append(self.model.get_nImmuneCells())
            self.cancerCells.append(self.model.get_nCancerCells())
            self.ax1.clear()
            self.ax1.plot(self.immuneCells, label="Immune")
            self.ax1.plot(self.cancerCells, label="Cancer")
            self.ax1.legend()


        
    def run(self):
        ani = FuncAnimation(self.fig, self.frame, frames=None, interval=100, repeat = False)
        plt.show()