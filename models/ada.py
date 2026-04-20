import pandas as pd
from strategies.model_strategy import ModelStrategy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import AdaBoostClassifier
from mlflow_log.mlflow_log import MlFlowLog
from models.xai.shap import ShapXAI
from pathlib import Path


class AdaModel(ModelStrategy, MlFlowLog, ShapXAI):
    def __init__(self):
        super().__init__(model=AdaBoostClassifier(),
                          experiment_name="AdaBoost Experiment",
                         param_grid={
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1]
        })
        self.mlflow_log = MlFlowLog(self.model, self.experiment_name, self.param_grid)
        self.final_model_path = Path("data/xai/ada_top_features.csv")

    def train(self,data: Path):
        print("Preparing data for AdaModel...")
        self.df = pd.read_csv(data)
        X, y = self.df.drop(['Class/ASD', 'result'], axis=1), self.df['Class/ASD']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
        print("Training AdaModel with cleaned data...")
        super().experiment()  # Call the MLflow logging experiment
        print("Evaluating AdaModel...")
    
    def explains_ada(self):
        print("Generating SHAP explanations for AdaModel...")
        shap_explainer = ShapXAI(model=self.fitted_model, final_model_path=self.final_model_path)
        shap_values = shap_explainer.explains(self.X_train)
        return shap_values