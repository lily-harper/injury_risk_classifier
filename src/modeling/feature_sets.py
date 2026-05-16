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
    "same_driver_action",
    "any_unknown_vehicle",
    "any_aggressive_or_careless",
    "any_failure_to_yield",
    "any_unknown_human_factor",
    "is_highway",
    "speed_limit_missing"
]

CATEGORICAL_FEATURES = [
    "road_description",
    "road_condition",
    "light_condition",
    "tu1_vehicle_type_binned",
    "tu2_vehicle_type_binned",
    "tu1_driver_action_binned",
    "tu2_driver_action_binned",
    "tu1_human_factor_binned",
    "tu2_human_factor_binned",
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

leakage_or_bad_cols = [
    "top_traffic_accident_offense",
    "first_occurrence_date",
    "date",
    "time",
    "datetime",
    "index_right",
    ]