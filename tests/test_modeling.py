import numpy as np
from sklearn.linear_model import LogisticRegression

def test_model_predict_proba():
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, size=10)

    model = LogisticRegression()
    model.fit(X, y)

    probs = model.predict_proba(X)

    assert probs.shape == (10, 2)
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)