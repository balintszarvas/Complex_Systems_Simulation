from model.classes.visualizer import Visualizer
from model.classes.cancerImmuneModel import CancerImmuneModel


def main():
    """Simple test function that runs the model in visualization mode with set parameters"""
    model = CancerImmuneModel(100, 100, 0.5, 0.4)
    model.seedImmune(round(100 * 100 / 10))
    model.seedCancer(1)

    vis = Visualizer(model)
    vis.run()
    for i in range(100):
        model.timestep()

    model.plot_cluster_sizes()
    model.plot_avalanche_sizes()
    equilibrium = model.get_equilibrium()
    print(f"The steady state is {equilibrium}")

if __name__ == "__main__":
    main()
