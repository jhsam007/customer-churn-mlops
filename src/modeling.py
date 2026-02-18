import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier


def tune_xgboost(X, y, preprocessor, n_trials: int, cv_folds: int):

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }

        model = XGBClassifier(**params)

        from sklearn.pipeline import Pipeline

        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        score = cross_val_score(
            pipeline,
            X,
            y,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
        ).mean()

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params


def get_model(model_type: str, params=None, random_state: int = 42):
    if model_type == "logistic":
        return LogisticRegression(max_iter=1000)

    if model_type == "xgboost":
        return XGBClassifier(
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            **(params or {})
        )

    raise ValueError("Unsupported model type")