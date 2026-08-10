import pandas as pd

from src.modeling.split import temporal_split

# test leakage 
def test_target_not_in_features():
    crashes = pd.DataFrame(
        {
            "date": ["2023-06-01", "2024-06-01", "2025-06-01"],
            "hour": [8, 12, 18],
            "injured": [0, 1, 0],
        }
    )

    X_train, y_train, X_val, y_val, X_test, y_test = temporal_split(
        crashes,
        features=["hour"],
        target_col="injured",
    )

    for features in (X_train, X_val, X_test):
        assert list(features.columns) == ["hour"]
        assert "injured" not in features.columns

    for target in (y_train, y_val, y_test):
        assert target.name == "injured"
