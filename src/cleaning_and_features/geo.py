import pandas as pd
import geopandas as gpd

def make_crash_points(
        df: pd.DataFrame,
        lon_col: str = "lon",
        lat_col: str = "lat") -> gpd.GeoDataFrame:
    out = df.copy()

    out = out.dropna(subset=[lat_col, lon_col])

    crashes_gdf = gpd.GeoDataFrame(
        out, 
        geometry=gpd.points_from_xy(out[lon_col], out[lat_col]),
        crs = "EPSG:4326"
    )

    return crashes_gdf

def convert_to_meters(crashes_gdf, roads_gdf):
    crashes_proj = crashes_gdf.to_crs("EPSG:26913")
    roads_proj = roads_gdf.to_crs("EPSG:26913")

    crashes_with_roads = gpd.sjoin_nearest(
        crashes_proj,
        roads_proj,
        how="left",
        distance_col="distance_to_road_m"
        )

    return crashes_with_roads

def match_quality(gdf, col):
    gdf = gdf.copy()

    gdf["road_join_quality"] = pd.cut(
        gdf[col],
        bins=[-1, 10, 25, 50, 100, float("inf")],
        labels=["excellent", "good", "okay", "weak", "bad"])
    
    gdf["road_match_ok"] = (gdf[col] <= 50)

    return gdf

def add_speed_limit_features(gdf, col):
    gdf = gdf.copy()

    gdf[col] = pd.to_numeric(
    gdf[col],
    errors="coerce")

    gdf["speed_limit_missing"] = (gdf[col].isna())

    gdf["speed_limit_category"] = pd.cut(
        gdf[col],
        bins=[0, 20, 25, 35, 45, 55, 80],
        labels=[
            "very_low",
            "low",
            "medium",
            "medium_high",
            "high",
            "very_high"
        ],
        include_lowest=True) 

    return gdf