from xml.parsers.expat import model
from models.xai.shap import ShapXAI
from strategies.model_strategy import ModelStrategy
from mlflow_log.mlflow_log import MlFlowLog
from xgboost import XGBClassifier   
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd


class XGBoostModel(ModelStrategy, MlFlowLog):
    def __init__(self):
        super().__init__(model=XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            error_score="raise"),
            experiment_name="XGBoost Experiment", param_grid={
            "n_estimators": [100, 300],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
    })
        self.mlflow_log = MlFlowLog(self.model, self.experiment_name, self.param_grid)
        self.final_model_path = Path("data/xai/xgb_top_features.csv")

    def train(self, data: Path):
        print("Training XGBoostModel with cleaned data...")
        self.df = pd.read_csv(data)
        X, y = self.df.drop('Class/ASD', axis=1), self.df['Class/ASD']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
        self.y_train = (self.y_train == "YES").astype(int)
        self.y_test = (self.y_test == "YES").astype(int)

        print("Training XGBoostModel with cleaned data...")
        super().experiment()
    
    def explains_xgb(self):
        print("Generating SHAP explanations for XGBoostModel...")
        shap_explainer = ShapXAI(model=self.fitted_model, final_model_path=self.final_model_path)
        shap_values = shap_explainer.explains(self.X_train)
        return shap_values
        
        