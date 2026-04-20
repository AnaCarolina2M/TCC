from sklearn.ensemble import AdaBoostClassifier

from strategies.xai import IXAI
import numpy as np
import pandas as pd
import shap     
from pathlib import Path

class ShapXAI(IXAI):
    def __init__(self, model, final_model_path: Path):
        self.fitted_model = model 
        if model is None:
            raise ValueError("Model must be trained before using SHAP")
        
        self.model_csv = final_model_path

    def explains(self, X_train):
        if self.fitted_model is None:
            raise ValueError("Model must be trained before using SHAP")
        try:
            if isinstance(self.fitted_model, AdaBoostClassifier):

                X_background = shap.sample(X_train, 100)
                explainer = shap.KernelExplainer(
                    self.fitted_model.predict_proba,   # 👈 THIS is the key fix
                    X_background
                )
                shap_values = explainer.shap_values(X_train, nsamples=100)
                values = shap_values[1] if isinstance(shap_values, list) else shap_values
                values = np.array(values)
                print("Values shape:", values.shape)
                importance = np.abs(values).mean(axis=(0, 2))
                print("Importance shape:", importance.shape)
                print("Columns:", len(X_train.columns))
                features = list(X_train.columns)  
                df = pd.DataFrame()
                df["feature"] = features
                df["importance"] = importance
                df =df.sort_values(by="importance", ascending=False )
                # 4. Save to CSV
                df.to_csv(self.model_csv, sep="\t", index=False)
                return shap_values
            else:
                print("Explaining model predictions using SHAP...")
                explainer = shap.Explainer(self.fitted_model, X_train)
                shap_values = explainer(X_train)
                importance = np.abs(shap_values).mean(axis=0)
                feature_importance = pd.DataFrame({
                    "feature": X_train.columns,
                    "importance": importance
                })
                print("Saving CSV now...")
                feature_importance = feature_importance.sort_values(by="importance", ascending=False)
                csv = feature_importance.to_csv(self.model_csv, sep="\t", index=False)
                return shap_values,csv
        except Exception as e:
            print(f"Error during SHAP explanation: {e}")
            return None