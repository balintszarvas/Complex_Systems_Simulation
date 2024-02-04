from model.classes.visualizer import Visualizer
from model.classes.bacteriaImmuneModel import bacteriaImmuneModel

"""Demo program for bacteria Immune model,

Runs the model in a visualizer with deault setting to check whether everything runs properly.
"""

# Default settings
LEN             = 200
FRAC_IMMUNE     = 0.0060
P_IMMUNEKILL    = 1
P_BACTERIAMULT  = 0.01
P_BACTERIASPAWN = 0.01


def main():
    """Simple test function that runs the model in visualization mode with set parameters"""
    model = bacteriaImmuneModel(LEN, LEN, P_IMMUNEKILL, P_BACTERIAMULT,P_BACTERIASPAWN)
    model.seedbacteria(100)

    model.seedImmune(round(LEN**2 * FRAC_IMMUNE))

    vis = Visualizer(model)
    vis.run()


if __name__ == "__main__":
    main()
