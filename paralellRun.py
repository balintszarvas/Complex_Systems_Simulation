import matplotlib.pyplot as plt
from model.classes.cancerImmuneModel import CancerImmuneModel

LEN = 200

def multiPlot(maxIter: int, runs: int):
        plots = []
        fig, ax = plt.subplots()
        assert(isinstance(ax, plt.Axes))

        for i in range(runs):
            model = CancerImmuneModel(LEN, LEN, 1.0, 0.01)
            model.seedImmune(round(LEN**2 * 0.010))

            print(i)
            vals = []
            for iter in range(maxIter):
                model.timestep()
                vals.append(model.get_nImmuneCells() / LEN**2)
        
            plots.append(vals)
        
        for i in range(runs):
            ax.plot(plots[i], label=i)
        
        plt.show()

multiPlot(5000, 5)