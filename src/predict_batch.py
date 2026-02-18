import argparse
import pandas as pd

from utils import load_model
from decision import apply_threshold


def predict_dataframe(pipeline, df: pd.DataFrame):
    proba = pipeline.predict_proba(df)[:, 1]
    pred = apply_threshold(proba)
    return proba, pred


def main(args):
    pipeline = load_model(args.model_path)
    df = pd.read_csv(args.input)

    proba, pred = predict_dataframe(pipeline, df)

    df["churn_probability"] = proba
    df["churn_prediction"] = pred

    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    main(args)