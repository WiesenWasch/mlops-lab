import os
import mlflow
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

def perform_nested_cv(pipeline, param_grid, X, y, outer_cv, inner_cv):
    """Performs nested cross-validation."""
    outer_scores = []
    chosen_params = []
    chosen_models = []

    # Outer Loop: Generalization Estimate
    for _fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # Inner Loop: Hyperparameter Tuning
        grid = GridSearchCV(pipeline, param_grid=param_grid, cv=inner_cv,
                            scoring='roc_auc', n_jobs=-1)
        grid.fit(X_tr, y_tr)

        # Evaluate on the independent Outer Test Set
        y_pred_proba = grid.best_estimator_.predict_proba(X_te)[:, 1]
        score = roc_auc_score(y_te, y_pred_proba)

        outer_scores.append(score)
        chosen_params.append(grid.best_params_)
        chosen_models.append(grid.best_estimator_)

    # Convert results for reporting
    # outer_scores = np.array(outer_scores)
    # mean_auc = outer_scores.mean()
    # std_auc = outer_scores.std()

    best_outer_score = max(outer_scores)
    best_index = outer_scores.index(best_outer_score)
    best_model = chosen_models[best_index]
    best_params = chosen_params[best_index]

    return best_outer_score, best_params, best_model

def train(data_path:str, outer_split:int=5, inner_split:int=3):
    data_bc = pd.read_csv(data_path)
    X_bc = data_bc.drop(columns=["target"])
    y_bc = data_bc["target"]

    mlflow.set_experiment(os.getenv("DATA_5750_EXPERIMENT_NAME", default="give_me_a_name"))
    with mlflow.start_run(run_name="model_training"):
        # setup Cross-Validation strategies
        RANDOM_STATE = 42
        outer_cv = StratifiedKFold(n_splits=outer_split, shuffle=True, random_state=RANDOM_STATE)
        inner_cv = StratifiedKFold(n_splits=inner_split, shuffle=True, random_state=RANDOM_STATE)

        # setup sklearn pipeline
        pipe_B = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(probability=True, random_state=RANDOM_STATE))
        ])
        # define hyperparameter search grid
        param_grid_B = {'clf__C': [0.1, 1, 10], 'clf__gamma': ['scale', 'auto']}

        # train model
        model_auc, chosen_params, best_model = perform_nested_cv(pipe_B, param_grid_B, X_bc, y_bc, outer_cv, inner_cv)

        # log results with mlflow
        mlflow.log_metric("auc", model_auc)
        mlflow.sklearn.log_model(best_model, name="model", input_example=X_bc)
        for param_name, param_value in chosen_params.items():
            mlflow.log_param(param_name, param_value)
        print(f"Model trained, AUC={model_auc:.4f}")
if __name__ == "__main__":
    train(
        data_path=os.getenv("DATA_5750_INGEST_DATA", default="data/bc_data.csv")
    )