import numpy as np
from pathlib import Path
import pandas as pd 
from src.cleaning_and_features import features 

from src.cleaning_and_features import clean
from src.paths import PROJECT_ROOT, CLEANED_DATA_PATH, BASE_FEATURES, import_data, save_data

def main():
    df = import_data(CLEANED_DATA_PATH, input = "parquet")
    df = features.add_vehicle_size_features(df)
    df = features.vehicle_size_actions(df)
    df = features.create_time_variables(df)
    df = features.create_highway_indicator(df)

    df = features.direction_conflict_type(df, 
                            col1 = "tu1_direction",
                            col2 = "tu2_direction")
    
    df = df.drop(columns=["tu1_direction", "tu2_direction",
                          "road_condition", "light_condition",
                          ])

    df = clean.convert_column_types(
        df, 
        bool_cols=[
            "weekend", "weekday",
                   "is_night",
                   "morning_rush",
                   "evening_rush",
                   ],
        category_cols = [
            "smaller_vehicle_action",
            "larger_vehicle_action",
            "direction_conflict",
            "road_condition_binned",
            "light_condition_binned",
            "road_description_binned",
            "vehicle_size_relation",
            "vehicle_type_match"
        ],
    )

    save_data(df, BASE_FEATURES, "parquet")

    print(f"Project root: {PROJECT_ROOT}")

if __name__ == "__main__":
    main()