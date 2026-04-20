from strategies.model_strategy import ModelStrategy
from mlflow_log.mlflow_log import MlFlowLog
from sklearn.ensemble import GradientBoostingClassifier   
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd


class GradientBoostingModel(ModelStrategy, MlFlowLog):
    def __init__(self):
        super().__init__(model=GradientBoostingClassifier(),
            experiment_name="GradientBoosting Experiment",
            param_grid={
            "n_estimators": [100, 300],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.1]
    })
        self.mlflow_log = MlFlowLog(self.model, self.experiment_name, self.param_grid)

    def train(self, data: Path):
        print("Preparing data for GradientBoostingModel...")
        self.df = pd.read_csv(data)
        X, y = self.df.drop('Class/ASD', axis=1), self.df['Class/ASD']
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
        y_train = (y_train == "YES").astype(int)
        y_test = (y_test == "YES").astype(int)

        print("Training GradientBoostingModel with cleaned data...")
        super().experiment(X_train, y_train, X_test, y_test)