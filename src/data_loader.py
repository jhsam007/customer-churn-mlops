import pandas as pd
from pathlib import Path


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded dataset is empty.")
    
    #Fix TotalCharges — coerce to float (some rows have empty strings)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    
    df = df.drop(columns=["customerID"], errors="ignore")


    return df