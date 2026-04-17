from matplotlib.pyplot import grid
from sklearn.model_selection import GridSearchCV
from strategies.mlflow_strategy import IMLFlowLog
from mlflow import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class MlFlowLog(IMLFlowLog):
    def __init__(self,model, experiment_name, param_grid: dict):
        self.model = model
        self.experiment_name = experiment_name
        self.param_grid = param_grid
        self.grid = None

    def experiment(self, X_train, y_train, X_test, y_test):
        print("Logging experiment details to MLflow...")
        mlflow.set_experiment(experiment_name=self.experiment_name)
        with mlflow.start_run():
            self.grid = GridSearchCV(self.model, self.param_grid, cv=5)
            self.grid.fit(X_train, y_train)
            y_pred = self.grid.predict(X_test)
            mlflow.log_params(self.grid.best_params_)
            mlflow.log_metric("best_cv_score", self.grid.best_score_)
            mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
            mlflow.sklearn.log_model(self.grid.best_estimator_, "model")
            mlflow.end_run()