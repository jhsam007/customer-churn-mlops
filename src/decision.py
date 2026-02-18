import numpy as np

def apply_threshold(
    y_proba: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Convert predicted probabilities into binary labels
    using a configurable decision threshold.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    return (y_proba >= threshold).astype(int)