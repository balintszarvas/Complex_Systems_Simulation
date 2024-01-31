from model.classes.visualizer import Visualizer
from model.classes.cancerImmuneModel import CancerImmuneModel

LEN         = 200
FRAC_IMMUNE = 0.0060

def main():
    """Simple test function that runs the model in visualization mode with set parameters"""
    model = CancerImmuneModel(LEN, LEN, 0.7, 0.0001, 0.01)
    model.seedCancer(100)
    
    #for i in range(100):
    #    model.timestep()
    
    model.seedImmune(round(LEN**2 * FRAC_IMMUNE))

    vis = Visualizer(model)
    vis.run()


if __name__ == "__main__":
    main()