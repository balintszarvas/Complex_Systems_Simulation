from model.classes.visualizer import Visualizer
from model.classes.cancerImmuneModel import CancerImmuneModel


def main():
    """Simple test function that runs the model in visualization mode with set parameters"""
    model = CancerImmuneModel(200, 200, 1.0, 0.01)
    model.seedCancer(5)

    for i in range(100):
        model.timestep()

    model.seedImmune(round(20 * 20 / 10))

    vis = Visualizer(model)
    vis.run()
    model.plot_cluster_sizes()

if __name__ == "__main__":
    main()
