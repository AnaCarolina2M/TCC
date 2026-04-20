from strategies.model_strategy import ModelStrategy
from mlflow_log.mlflow_log import MlFlowLog
from sklearn.ensemble import GradientBoostingClassifier 
from models.xai.shap import ShapXAI  
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd


class GradientBoostingModel(ModelStrategy, MlFlowLog,ShapXAI):
    def __init__(self):
        super().__init__(model=GradientBoostingClassifier(),
            experiment_name="GradientBoosting Experiment",
            param_grid={
            "n_estimators": [100, 300],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.1]
    })
        self.mlflow_log = MlFlowLog(self.model, self.experiment_name, self.param_grid)
        self.final_model_path = Path("data/xai/gbc_top_features.csv")

    def train(self, data: Path):
        print("Preparing data for GradientBoostingModel...")
        self.df = pd.read_csv(data)
        X, y = self.df.drop(['Class/ASD', 'result'], axis=1), self.df['Class/ASD']
        X = X.astype('float64')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
        self.y_train = (self.y_train == "YES").astype(int)
        self.y_test = (self.y_test == "YES").astype(int)
        print("Training GradientBoostingModel with cleaned data...")
        super().experiment()
    
    def explains_gbc(self):
        print("Generating SHAP explanations for GradientBoostingModel...")
        shap_explainer = ShapXAI(model=self.fitted_model, final_model_path=self.final_model_path)
        shap_values = shap_explainer.explains(self.X_train)
        return shap_values