from pipeline.cleaning import CleaningPipeline
from pathlib import Path
from models.ada import AdaModel
# from models.model2 import ABCModel

class Experiment():
    def __init__(self, data: Path):
        self.data = data
        self.cleaner = CleaningPipeline(self.data)
        self.cleaner.file_path = self.data
        self.cleaner.output_path = Path("data/cleaned_data.csv")
        self.ada_model = AdaModel()  # Pass the path to the cleaned data
        # self.model2 = ABCModel()

    def run(self):
        # Step 1: Clean the data
        self.cleaned_data = self.cleaner.cleans()
        self.ada_model.train(self.cleaner.output_path)  # Train the AdaModel with the cleaned data

        # Step 2: Train Model 1
        # self.model1.train(self.cleaned_data)

        # Step 3: Train Model 2
        # self.model2.train(self.cleaned_data)

        # Additional steps like evaluation can be added here