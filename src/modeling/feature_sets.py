# src/modeling/feature_sets.py

NUMERIC_FEATURES = [
    "hour",
    "month",
    "speed_limit"
]

BOOLEAN_FEATURES = [
    "is_night",
    "morning_rush",
    "evening_rush",
    "is_highway",
    "speed_limit_missing"
]

CATEGORICAL_FEATURES = [
    "road_description_binned",
    "road_condition_binned",
    "light_condition_binned",
    "tu1_vehicle_type_binned",
    "tu2_vehicle_type_binned",
    "tu1_driver_action_binned",
    "tu2_driver_action_binned",
    "tu1_human_fac_binned",
    "tu2_human_fac_binned",
    "vehicle_type_match",
    "vehicle_size_relation",
    "smaller_vehicle_action",
    "larger_vehicle_action",
    "direction_conflict",
    "district_id",
    "road_join_quality",
    "speed_limit_category",
]

FEATURE_SETS = {
    "numeric": NUMERIC_FEATURES,
    "boolean": BOOLEAN_FEATURES,
    "categorical": CATEGORICAL_FEATURES,
}

MODEL_FEATURES = (
    NUMERIC_FEATURES
    + BOOLEAN_FEATURES
    + CATEGORICAL_FEATURES
)

LEAKAGE_OR_BAD_COLS = [
    "top_traffic_accident_offense",
    "first_occurrence_date",
    "date",
    "time",
    "datetime",
    "index_right",
    "road_name",
    "incident_address_clean",
    ]
