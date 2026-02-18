import pandas as pd
import scipy.sparse
from src.preprocessing import build_preprocessor

def test_preprocessor_output_shape():
    df = pd.DataFrame({
        "gender": ["Male", "Female"],
        "SeniorCitizen": [0, 1],
        "tenure": [5, 10],
        "MonthlyCharges": [50.0, 80.0],
        "TotalCharges": [250.0, 800.0],
        "Contract": ["Month-to-month", "Two year"],
        "Churn": ["Yes", "No"]  # ✅ add target column
    })

    preprocessor = build_preprocessor(df, target_col="Churn")  # ✅ add target_col
    X = df.drop(columns=["Churn"])
    X_transformed = preprocessor.fit_transform(X)

    assert X_transformed.shape[0] == 2

def test_no_nan_after_preprocessing():
    df = pd.DataFrame({
        "gender": ["Male"],
        "SeniorCitizen": [0],
        "tenure": [5],
        "MonthlyCharges": [50.0],
        "TotalCharges": [250.0],
        "Contract": ["Month-to-month"],
        "Churn": ["Yes"]  # add target column
    })

    preprocessor = build_preprocessor(df, target_col="Churn")  # add target_col
    X = df.drop(columns=["Churn"])
    X_transformed = preprocessor.fit_transform(X)

    # handle sparse matrix
    if scipy.sparse.issparse(X_transformed):
        X_transformed = X_transformed.toarray()

    assert not pd.isnull(X_transformed).any()