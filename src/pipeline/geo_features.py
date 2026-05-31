import geopandas as gpd
import pandas as pd 
from pathlib import Path

from src.cleaning_and_features import geo, clean
from src.paths import BASE_FEATURES, COOR_PATH, SAMPLE_DATA_PATH, MODELING_DATA, import_data, save_data, ALL_DATA


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

    crashes_with_roads = clean.convert_column_types(
        crashes_with_roads, 
        bool_cols=["is_highway"],
        category_cols = 
            ["road_join_quality",
            "speed_limit_category",
            "road_description_binned"
        ],
        int_cols=["speed_limit"]
    )

    mod = crashes_with_roads.drop(columns=[
        "datetime", "time", "incident_address_clean",
        "road_name", "lat", "lon", "distance_to_road_m",
        "road_match_ok", "top_traffic_accident_offense",
        ])

    return crashes_with_roads, mod

def main():
    df = import_data(BASE_FEATURES, "parquet")
    speed_df, mod = speed_data(df, road_path=COOR_PATH)

    save_data(speed_df, ALL_DATA, "parquet") 
    print(f"Saved {speed_df.shape[0]} rows and {speed_df.shape[1]} columns to {ALL_DATA}")

    save_data(mod, MODELING_DATA, "parquet") 
    print(f"Saved {mod.shape[0]} rows and {mod.shape[1]} columns to {MODELING_DATA}")

    sample = speed_df.sample(50, random_state = 67)
    save_data(sample, SAMPLE_DATA_PATH, "csv")
    print(f"Saved {sample.shape[0]} rows and {sample.shape[1]} columns to {SAMPLE_DATA_PATH}")


if __name__ == "__main__":
    main()