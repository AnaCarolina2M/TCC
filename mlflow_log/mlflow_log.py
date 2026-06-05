from matplotlib.pyplot import grid
from sklearn.model_selection import GridSearchCV
from strategies.mlflow_strategy import IMLFlowLog
from mlflow import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,confusion_matrix,ConfusionMatrixDisplay,RocCurveDisplay

class MlFlowLog(IMLFlowLog):
    def __init__(self,model, experiment_name, param_grid: dict):
        self.model = model
        self.experiment_name = experiment_name
        self.param_grid = param_grid
        self.grid = None
        self.fitted_model = None
        self.X_train = None
        self.y_train = None 
        self.X_test = None
        self.y_test = None

    #def experiment(self, X_train, y_train, X_test, y_test):
    def experiment(self):
        print("Logging experiment details to MLflow...")
        mlflow.set_experiment(experiment_name=self.experiment_name)
        with mlflow.start_run():
            self.grid = GridSearchCV(self.model, self.param_grid, cv=5)
            self.fitted_model = self.grid.fit(self.X_train, self.y_train)
            self.fitted_model = self.grid.best_estimator_
            print(type(self.fitted_model))
            y_pred = self.grid.predict(self.X_test)
            mlflow.log_params(self.grid.best_params_)
            mlflow.log_metric("best_cv_score", self.grid.best_score_)
            mlflow.log_metric("accuracy", accuracy_score(self.y_test, y_pred))
            # mlflow.log_metric("f1_score", f1_score(self.y_test, y_pred))
            y_proba = self.grid.predict_proba(self.X_test)[:, 1]
            auc = roc_auc_score(self.y_test,y_proba)
            mlflow.log_metric("roc_auc",auc)
            mlflow.sklearn.log_model(self.grid.best_estimator_, "model")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(self.y_test,y_pred,ax=ax)
            plt.tight_layout()
            mlflow.log_figure(fig,"confusion_matrix.png")
            plt.close()
            fig, ax = plt.subplots()
            RocCurveDisplay.from_predictions(self.y_test,y_proba, pos_label='YES',ax=ax)
            plt.tight_layout()
            mlflow.log_figure(fig,"roc_curve.png")
            plt.close()
            mlflow.sklearn.log_model(
                self.fitted_model,
                "model")
            mlflow.end_run()
            return self.fitted_model