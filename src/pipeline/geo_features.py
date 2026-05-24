import geopandas as gpd
import pandas as pd 
from pathlib import Path

from src.cleaning_and_features import geo, clean
from src.paths import BASE_FEATURES, COOR_PATH, SAMPLE_DATA_PATH, MODELING_DATA, import_data


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
    df = import_data(BASE_FEATURES)
    speed_df = speed_data(df, road_path=COOR_PATH)

    speed_df = clean.convert_column_types(
        speed_df, 
        bool_cols=["is_highway"],
        category_cols = 
            ["road_join_quality",
            "speed_limit_category"
        ],
        int_cols=["speed_limit"]
    )

    speed_df.to_parquet(MODELING_DATA, index = False)
    print(f"Saved {speed_df.shape[0]} rows and {speed_df.shape[1]} columns to {MODELING_DATA}")

    sample = speed_df.sample(50, random_state = 67)
    sample.to_csv(SAMPLE_DATA_PATH, index = False)
    print(f"Saved {sample.shape[0]} rows and {sample.shape[1]} columns to {SAMPLE_DATA_PATH}")


if __name__ == "__main__":
    main()