import shap
import pandas as pd

def compute_shap_values(model, X: pd.DataFrame):
    """
    Compute SHAP values for trained model.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    return shap_values


def get_global_feature_importance(shap_values):
    """
    Return mean absolute SHAP values for global importance.
    """
    return abs(shap_values.values).mean(axis=0)