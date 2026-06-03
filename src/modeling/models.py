from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import PredefinedSplit, GridSearchCV

import numpy as np
import pandas as pd

from src.modeling.preprocessors import build_preprocessor, tree_prep, nb_preprocessor
from src.modeling.feature_sets import FEATURE_SETS

def build_dummy_model(constant):
    return Pipeline(steps = [
        ("preprocessor", build_preprocessor()),
        ("model", DummyClassifier(strategy= "constant", constant=constant))
    ])

def build_logistic_model(class_weight, C=1.0):
    return Pipeline(steps = [
        ("preprocessor", build_preprocessor(FEATURE_SETS)),
        ("model", LogisticRegression(
            max_iter = 1000,
            class_weight= class_weight,
            random_state=67
        ))
    ])

def build_base_tree_model(class_weight):
    return Pipeline(steps = [
        ("preprocess", tree_prep()),
        ("tree", DecisionTreeClassifier(
            criterion="entropy",
            class_weight = class_weight,
            random_state=67,
        ))
    ])

def build_naive_bayes():
    return Pipeline(steps = [
        ("preprocessor", nb_preprocessor()),
        ("model", MultinomialNB())
    ])

def build_tuned_tree(X_train, X_val, y_train, y_val):
    tree_pipe = build_base_tree_model()

    param_grid = {
        "tree__criterion": ["gini", "entropy"],
        "tree__max_depth": [3, 5, 6, 8, 10, None],
        "tree__min_samples_leaf": [10, 25, 50, 100, 200],
        "tree__min_samples_split": [25, 50, 100, 200],
        "tree__class_weight": [None, "balanced"],
        } 

    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)

    test_fold = np.concatenate([
        np.full(len(X_train), -1),  # -1 means always train
        np.zeros(len(X_val))        # 0 means validation fold
        ])
    
    grid = GridSearchCV(
        estimator=tree_pipe,
        param_grid=param_grid,
        scoring="recall",  
        cv= test_fold,
        n_jobs=-1,
        refit=True,
        verbose=1
        )
    
    return grid.best_estimator_
    
def positive_class_proba(model, X, positive_label=1):
    class_index = list(model.classes_).index(positive_label)
    return model.predict_proba(X)[:, class_index]
