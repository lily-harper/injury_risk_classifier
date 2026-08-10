import numpy as np
from sklearn.linear_model import LogisticRegression

from src.modeling.models import positive_class_proba


def test_positive_class_probabilities_are_valid():
    X_train = np.array([[0], [1], [2], [3]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[0.5], [1.5], [2.5]])
    model = LogisticRegression().fit(X_train, y_train)

    probabilities = positive_class_proba(model, X_test)

    assert probabilities.shape == (len(X_test),)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)

def classify_probability(probability, threshold):
    return int(probability > threshold)

def test_probability_threshold_positive():
    assert classify_probability(.50, threshold = .48) == 1

def test_probability_threshold_negative():
    assert classify_probability(.2, threshold = 48) == 0
