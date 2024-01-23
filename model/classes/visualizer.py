from .cancerImmuneModel import CancerImmuneModel
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Visualizer:
    def __init__(self, model: CancerImmuneModel) -> None:
        self.model = model
        fig, ax = plt.subplots()
        assert isinstance(ax, plt.Axes)

        self.fig = fig
        self.ax = ax
        self.fig.show()
    
    def frame(self, i):
        if self.model.get_nCancerCells() > 0:
            self.model.timestep()
            self.ax.clear()
            self.ax.imshow(self.model.cancerLattice + self.model.immuneLattice)
        
    def run(self):
        ani = FuncAnimation(self.fig, self.frame, frames=None, interval=100, repeat = False)
        plt.show()