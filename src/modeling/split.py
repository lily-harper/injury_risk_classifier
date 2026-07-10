from src.modeling.feature_sets import MODEL_FEATURES

import pandas as pd 

TRAIN_END_YEAR = 2023
VALIDATION_YEAR = 2024
TEST_START_YEAR = 2025


def split_config() -> dict:
    return {
        "train_end_year": TRAIN_END_YEAR,
        "validation_year": VALIDATION_YEAR,
        "test_start_year": TEST_START_YEAR,
    }


def split_summary(sets, target_col) -> pd.DataFrame:
    rows = []

    for split_name, split_df in sets.items():
        counts = split_df[target_col].value_counts(dropna=False)
        rates = split_df[target_col].value_counts(normalize=True, dropna=False)

        for class_value in counts.index:
            rows.append({
                "subset": split_name,
                "class": class_value,
                "class_count": counts[class_value],
                "class_rate": round(rates[class_value], 4),
                "total_observations": len(split_df),
            })

    return pd.DataFrame(rows)


def temporal_split(
    df,
    features=MODEL_FEATURES,
    target_col="injured",
    date_col="date",
    summary=False,
    train_end_year=TRAIN_END_YEAR,
    validation_year=VALIDATION_YEAR,
    test_start_year=TEST_START_YEAR,
):
    df = df[features + [target_col, date_col]].copy()

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    train = df[df[date_col].dt.year <= train_end_year]
    validate = df[df[date_col].dt.year == validation_year]
    test = df[df[date_col].dt.year >= test_start_year]

    X_train = train[features]
    y_train = train[target_col]

    X_val = validate[features]
    y_val = validate[target_col]

    X_test = test[features]
    y_test = test[target_col]
    if summary:
        report = split_summary(
            sets={
                "train": train,
                "validation": validate,
                "test": test,
            },
            target_col=target_col,
        )
        return X_train, y_train, X_val, y_val, X_test, y_test, report

    return X_train, y_train, X_val, y_val, X_test, y_test
