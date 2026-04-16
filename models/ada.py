import pandas as pd
from strategies.model_strategy import ModelStrategy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import AdaBoostClassifier
from mlflow_log.mlflow_log import MlFlowLog
from pathlib import Path


class AdaModel(ModelStrategy, MlFlowLog):
    def __init__(self):
        super().__init__(model=AdaBoostClassifier(),
                          experiment_name="AdaBoost Experiment",
                         param_grid={
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1]
        })
        self.mlflow_log = MlFlowLog(self.model, self.experiment_name, self.param_grid)

    def train(self,data: Path):
        self.df = pd.read_csv(data)
        X, y = self.df.drop('Class/ASD', axis=1), self.df['Class/ASD']
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
        print("Training AdaModel with cleaned data...")
        super().experiment(X_train, y_train, X_test, y_test)