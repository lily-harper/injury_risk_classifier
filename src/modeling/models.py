from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

import numpy as np
import pandas as pd
from pathlib import Path

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

def build_tree_model(
    criterion="entropy",
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=2,
    class_weight=None,
):
    return Pipeline(steps=[
        ("preprocess", tree_prep()),
        ("tree", DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=67,
        )),
    ])

def build_naive_bayes():
    return Pipeline(steps = [
        ("preprocessor", nb_preprocessor()),
        ("model", MultinomialNB())
    ])
    
def positive_class_proba(model, X, positive_label=1):
    class_index = list(model.classes_).index(positive_label)
    return model.predict_proba(X)[:, class_index]
