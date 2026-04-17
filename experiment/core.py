from pipeline.cleaning import CleaningPipeline
from pathlib import Path
from models.ada import AdaModel
from models.xg_boost import XGBoostModel
from models.gbc import GradientBoostingModel

class Experiment():
    def __init__(self, data: Path):
        self.data = data
        self.cleaner = CleaningPipeline(self.data)
        self.cleaner.file_path = self.data
        self.cleaner.output_path = Path("data/cleaned_data.csv")
        self.ada_model = AdaModel()  # Pass the path to the cleaned data
        self.xgboost_model = XGBoostModel()
        self.gradient_boosting_model = GradientBoostingModel()

    def run(self):
        print('Starting the experiment...')
        self.cleaned_data = self.cleaner.cleans()
        print('Data cleaning completed. Cleaned data saved to:', self.cleaner.output_path)
        self.ada_model.train(self.cleaner.output_path) 
        print('AdaBoostModel training completed.')
        self.xgboost_model.train(self.cleaner.output_path)  # Train the XGBoostModel with the cleaned data
        print('XGBoostModel training completed.')
        self.gradient_boosting_model.train(self.cleaner.output_path)  # Train the GradientBoostingModel with the cleaned data
        print('GradientBoostingModel training completed.')

        # Step 2: Train Model 1
        # self.model1.train(self.cleaned_data)

        # Step 3: Train Model 2
        # self.model2.train(self.cleaned_data)

        # Additional steps like evaluation can be added here