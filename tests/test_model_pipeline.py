import pandas as pd

from src.modeling.feature_sets import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
)
from src.modeling.models import build_logistic_model


def test_logistic_pipeline_fits_and_predicts_probabilities():
    row_count = 6
    model_data = pd.DataFrame(
        {
            **{column: range(row_count) for column in NUMERIC_FEATURES},
            **{
                column: [False, True, False, True, False, True]
                for column in BOOLEAN_FEATURES
            },
            **{
                column: ["category_a", "category_b"] * 3
                for column in CATEGORICAL_FEATURES
            },
        }
    )
    model_data[BOOLEAN_FEATURES] = model_data[BOOLEAN_FEATURES].astype("boolean")
    target = pd.Series([0, 1, 0, 1, 0, 1], name="injured")
    model = build_logistic_model(class_weight="balanced")

    model.fit(model_data[MODEL_FEATURES], target)
    probabilities = model.predict_proba(model_data[MODEL_FEATURES])

    assert probabilities.shape == (row_count, 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert probabilities.sum(axis=1).round(10).tolist() == [1.0] * row_count
