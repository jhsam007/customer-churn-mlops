import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import (
    data_config,
    model_config,
    training_config,
    artifact_config,
)
from data_loader import load_data
from preprocessing import build_preprocessor
from modeling import get_model, tune_xgboost
from evaluation import evaluate_classification
from decision import apply_threshold
from utils import save_model


def run_training_pipeline():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(artifact_config.mlflow_experiment)

    with mlflow.start_run():

        mlflow.log_param("model_type", model_config.model_type)


        df = load_data(data_config.raw_data_path)

        X = df.drop(columns=[data_config.target_col])
        y = df[data_config.target_col].map({"Yes": 1, "No": 0})

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=training_config.test_size,
            random_state=model_config.random_state,
            stratify=y,
        )

        preprocessor = build_preprocessor(df, data_config.target_col)

        if model_config.model_type == "xgboost":
            best_params = tune_xgboost(
                X_train,
                y_train,
                preprocessor,
                training_config.n_trials,
                training_config.cv_folds,
            )

            for k, v in best_params.items():
                mlflow.log_param(k, v)

            model = get_model("xgboost", best_params)
        else:
            model = get_model("logistic")

        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics, cm = evaluate_classification(y_test, y_proba)


        print("Metrics before logging:", metrics)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        model_path = artifact_config.model_dir / "churn_pipeline.pkl"
        save_model(pipeline, model_path)
        mlflow.log_artifact(str(model_path))

        return metrics


if __name__ == "__main__":
    run_training_pipeline()