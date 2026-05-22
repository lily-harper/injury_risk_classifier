import pandas as pd 
import geopandas as gpd
import numpy as np
from pathlib import Path

from src.cleaning_and_features import features 
from src.cleaning_and_features import geo
from src.cleaning_and_features import clean
from src.paths import MODELING_DATA, PROJECT_ROOT, COOR_PATH, PROCESSED_DATA_DIR, SAMPLE_DATA_PATH, CLEANED_DATA_PATH


def import_data(raw_data_path: Path) -> pd.DataFrame:
    if not raw_data_path.exists():
        raise FileNotFoundError(f"File not found: {raw_data_path}")
    
    return pd.read_parquet(raw_data_path) 

def speed_data(df:pd.DataFrame, road_path: Path) -> pd.DataFrame:
    roads_gdf = gpd.read_file(road_path)
    roads_gdf = roads_gdf[["SPEEDLIMIT", "FULLNAME", "geometry"]]

    crashes_gdf = geo.make_crash_points(df)

    crashes_with_roads = geo.convert_to_meters(crashes_gdf, roads_gdf)
    crashes_with_roads = geo.match_quality(crashes_with_roads, "distance_to_road_m")

    crashes_with_roads = crashes_with_roads.rename(columns={
        "SPEEDLIMIT": "speed_limit",
        "FULLNAME":"road_name"
        })
    
    crashes_with_roads = geo.add_speed_limit_features(crashes_with_roads, "speed_limit")

    crashes_with_roads = crashes_with_roads.drop(columns=["incident_address", "geometry"])

    return crashes_with_roads

def main():
    df = import_data(CLEANED_DATA_PATH)
    df = features.add_vehicle_size_features(df)
    df = features.vehicle_size_actions(df)
    df = features.create_time_variables(df)
    df = features.create_highway_indicator(df)

    df = features.direction_conflict_type(df, 
                            col1 = "tu1_direction",
                            col2 = "tu2_direction")
    
    df = df.drop(columns=["tu1_direction", "tu2_direction",
                          "road_condition", "light_condition"])

    speed_df = speed_data(df, road_path=COOR_PATH)

    df = clean.convert_column_types(
        df, 
        bool_cols=[
            "weekend", "weekday",
                   "is_night",
                   "morning_rush",
                   "evening_rush",
                   "is_highway"],
        category_cols = [
            "smaller_vehicle_action",
            "larger_vehicle_action",
            "direction_conflict",
            "road_join_quality",
            "speed_limit_category"
        ],
        int_cols=["speed_limit"]
    )

    speed_df.to_parquet(MODELING_DATA)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Saved {speed_df.shape[0]} rows and {speed_df.shape[1]} columns to {MODELING_DATA}")

    sample = speed_df.sample(50)
    sample.to_csv(SAMPLE_DATA_PATH, index = False)
    print(f"Saved {sample.shape[0]} rows and {sample.shape[1]} columns to {SAMPLE_DATA_PATH}")

if __name__ == "__main__":
    main()