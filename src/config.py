from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DataConfig:
    raw_data_path: Path = PROJECT_ROOT / "data/raw/Telco-Customer-Churn.csv"
    target_col: str = "Churn"


@dataclass
class ModelConfig:
    model_type: str = "xgboost"  # "logistic" | "xgboost"
    random_state: int = 42


@dataclass
class TrainingConfig:
    test_size: float = 0.2
    n_trials: int = 30
    cv_folds: int = 5


@dataclass
class EvaluationConfig:
    decision_threshold: float = 0.4


@dataclass
class ArtifactConfig:
    model_dir: Path = PROJECT_ROOT / "models"
    mlflow_experiment: str = "customer_churn_experiments"
    model_registry_name: str = "customer_churn_model"


# Instantiate configs
data_config = DataConfig()
model_config = ModelConfig()
training_config = TrainingConfig()
evaluation_config = EvaluationConfig()
artifact_config = ArtifactConfig()