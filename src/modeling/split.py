from src.modeling.feature_sets import MODEL_FEATURES

import pandas as pd 

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
    summary = False
):
    test_year=2023
    validation_year=2024
    train_end_year=2022

    df = df[features + [target_col, date_col]].copy()

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    train = df[df[date_col].dt.year <= train_end_year]
    test = df[df[date_col].dt.year == test_year]
    validate = df[df[date_col].dt.year == validation_year]

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
                "test": test,
                "validation": validate,
            },
            target_col=target_col,
        )
        return X_train, y_train, X_val, y_val, X_test, y_test, report

    return X_train, y_train, X_val, y_val, X_test, y_test
