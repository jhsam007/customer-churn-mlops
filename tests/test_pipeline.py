from src.pipeline import run_training_pipeline
from src.config import artifact_config

def test_pipeline_runs(tmp_path):
    results = run_training_pipeline()

    # Check metrics returned
    assert results is not None
    assert "roc_auc" in results
    assert "f1" in results
    assert "accuracy" in results

    # Check churn_pipeline.pkl was saved
    model_path = artifact_config.model_dir / "churn_pipeline.pkl"
    assert model_path.exists()