from strategies.xai import IXAI
import shap     

class ShapXAI(IXAI):
    def __init__(self, model):
        self.model = model
        self.explainer = None

    def explains(self, X_train):
        print("Explaining model predictions using SHAP...")
        self.explainer = shap.Explainer(self.model, X_train)
        shap_values = self.explainer(X_train)
        return shap_values