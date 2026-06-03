from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, FunctionTransformer

from src.modeling.feature_sets import FEATURE_SETS

def build_preprocessor(feature_sets=FEATURE_SETS):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    boolean_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_sets["numeric"]),
            ("bool", boolean_transformer, feature_sets["boolean"]),
            ("cat", categorical_transformer, feature_sets["categorical"]),
        ]
    )

    return preprocessor

def nb_preprocessor(feature_sets=FEATURE_SETS):
    numeric_nb = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
    ])

    categorical_nb = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    boolean_nb = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("to_int", FunctionTransformer(lambda X: X.astype(int)))
    ])

    preprocessor_nb = ColumnTransformer(
        transformers=[
            ("num", numeric_nb, feature_sets["numeric"]),
            ("bool", boolean_nb, feature_sets["boolean"]),
            ("cat", categorical_nb, feature_sets["categorical"]),
        ]
    )

    return preprocessor_nb

def tree_prep(feature_sets=FEATURE_SETS):
    numeric_tree = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_tree = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    boolean_tree = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor_tree = ColumnTransformer(
        transformers=[
            ("num", numeric_tree, feature_sets["numeric"]),
            ("bool", boolean_tree, feature_sets["boolean"]),
            ("cat", categorical_tree, feature_sets["categorical"]),
        ]
    )

    return preprocessor_tree
