from .core import Experiment
from pathlib import Path

if __name__ == "__main__":
    data = Path("data/Autism_Adult_Data.csv")  # Update this path to your dataset
    experiment = Experiment(data)
    experiment.run()