import pandas as pd

from src.cleaning_and_features.features import add_vehicle_size_features


def test_add_vehicle_size_features():
    crashes = pd.DataFrame(
        {
            "tu1_vehicle_type_binned": ["passenger", "unknown"],
            "tu2_vehicle_type_binned": ["bus", "suv"],
        }
    )

    result = add_vehicle_size_features(crashes)

    assert result.loc[0, "tu1_vehicle_size"] == 2
    assert result.loc[0, "tu2_vehicle_size"] == 4
    assert result.loc[0, "vehicle_type_match"] == "different"
    assert result.loc[0, "vehicle_size_diff"] == -2
    assert result.loc[0, "abs_vehicle_size_diff"] == 2
    assert result.loc[0, "vehicle_size_relation"] == "vehicle_2_larger"

    assert pd.isna(result.loc[1, "tu1_vehicle_size"])
    assert result.loc[1, "vehicle_type_match"] == "unknown"
    assert result.loc[1, "vehicle_size_relation"] == "unknown"

    assert "tu1_vehicle_size" not in crashes.columns
