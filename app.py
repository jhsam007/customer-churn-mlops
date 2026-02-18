from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from src.decision import apply_threshold
from src.config import evaluation_config, artifact_config

# Paths
MODEL_PATH = artifact_config.model_dir / "churn_pipeline.pkl"

# ✅ Load pipeline correctly
try:
    pipeline = joblib.load(MODEL_PATH)
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")

# FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0"
)

# Input schema
class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Health check
@app.get("/health")
def health():
    return {"status": "ok"}

# Prediction endpoint
@app.post("/predict")
def predict_churn(data: CustomerInput):
    try:
        # Convert request → DataFrame
        df = pd.DataFrame([data.model_dump()])

        # ✅ Use full pipeline directly (handles preprocessing internally)
        churn_prob = pipeline.predict_proba(df)[0, 1]

        # Apply configurable threshold
        threshold = evaluation_config.decision_threshold
        churn_label = int(
            apply_threshold(np.array([churn_prob]), threshold)[0]
        )

        return {
            "churn_probability": round(float(churn_prob), 4),
            "decision_threshold": threshold,
            "churn_label": churn_label
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))