from .cancerImmuneModel import CancerImmuneModel
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    
    def frame(self, i):
            self.model.timestep()
            self.ax0.clear()
            self.ax0.imshow(self.model.cancerLattice + self.model.immuneLattice)


        
    def run(self):
        ani = FuncAnimation(self.fig, self.frame, frames=None, interval=100, repeat = False)
        plt.show()