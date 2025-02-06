import json
import os

import numpy as np
import openml
from sklearn.model_selection import train_test_split

from optuna_tune import tune_hyper_parameters
from talent_classifier import DeepClassifier

if __name__ == "__main__":
    model = "modernNCA"
    e = DeepClassifier(model_type=model)
    dataset = openml.datasets.get_dataset(
        3,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    qualities = dataset.qualities
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=100, random_state=0, shuffle=True, stratify=y
    )

    config_opt_path = os.path.join(
        "LAMDA-TALENT",
        "LAMDA_TALENT",
        "configs",
        "opt_space",
        e.model_type + ".json",
    )
    with open(config_opt_path, "r") as file:
        opt_space = json.load(file)
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=0
    )
    tune_hyper_parameters(
        e,
        opt_space,
        X_train_sub,
        y_train_sub,
        X_val,
        y_val,
        categorical_indicator,
    )
