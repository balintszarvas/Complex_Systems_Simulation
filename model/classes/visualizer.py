from .bacteriaImmuneModel import bacteriaImmuneModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from typing import List

class Visualizer:
    """Real time visualizer for bacteriaImmuneModel

    Initialize object, then call Visualizer.run()
    
    Properties:
        model (bacterialImmuneModel): The model object to be visualized in real time
        laticceCells           (int): The total amount of cells on the lattice for occupancy calculations.
        fig             (plt.Figure): Figure object of the visualizer
        ax0               (plt.Axes): Axes object for the lattice visualization
        ax1               (plt.Axes): Axes object for occupancy plotting
        immuneCells    (List[float]): Immune cell occupancy per iteration.
        bacteriaCells  (List[float]): Bacteria cell occupancy per iteration.
    """
    def __init__(self, model: bacteriaImmuneModel) -> None:
        """
        Initializer function

        Args:
            model (bacterialImmuneModel): The model object to be visualized in real time.        
        """
        self.model = model
        self.latticeCells = self.model.dim[0] * self.model.dim[1]

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
        self.bacteriaCells: List[int] = []

    def frame(self, i):
        """
        Per frame function for visualizer. Advances the model by one timestep and updates ax0 and ax1.

        Args:
            i (int): matplotlib framenumber, required for funcAnimation call. (unused)
        """
        self.model.timestep()
        self.ax0.clear()
        self.ax0.imshow(self.model.bacteriaLattice * np.invert(self.model.immuneLattice.astype(bool)) + self.model.immuneLattice)

        self.immuneCells.append(self.model.get_nImmuneCells() / self.latticeCells)
        self.bacteriaCells.append(self.model.get_nbacteriaCells() / self.latticeCells)
        self.ax1.clear()
        self.ax1.plot(self.immuneCells,   label="Immune")
        self.ax1.plot(self.bacteriaCells, label="Bacteria")
        self.ax1.legend()

    def run(self):
        """
        Starts indefinite loop of visualizer frames until matplotlib window is closed.
        """
        ani = FuncAnimation(self.fig, self.frame, frames=None, interval=1, repeat = False)
        plt.show()
        # ani.save("animation_plot.gif", writer='pillow')