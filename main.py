from model.classes.visualizer import Visualizer
from model.classes.cancerImmuneModel import CancerImmuneModel


def main():
    """Simple test function that runs the model in visualization mode with set parameters"""
    model = CancerImmuneModel(200, 200, 1.0, 0.01)
    model.seedCancer(1)
    for i in range(500): # Unhindered growth for cancer cell
        model.timestep()
    model.seedImmune(round(200 * 200 / 150))

    vis = Visualizer(model)
    vis.run()


if __name__ == "__main__":
    main()