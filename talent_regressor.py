from typing import List
from unittest.mock import patch

import numpy as np
import openml
from sklearn.base import RegressorMixin
from sklearn.model_selection import train_test_split

from data_loader import (
    split_train_val,
    convert_test,
    generate_info,
)
from talent_classifier import (
    DeepClassifier,
    SuppressPrint,
)
from TALENT.model.utils import get_method


class DeepRegressor(DeepClassifier, RegressorMixin):
    def fit(self, X, y, categorical_indicator: List[bool]):
        """
        Fit the deep learning regression model.
        Parameters:
            X: Features (pre-split training data)
            y: Continuous target variable
            categorical_indicator: List indicating which features are categorical
        Returns:
            self
        """
        # Split the training data into train and validation
        self.categorical_indicator = categorical_indicator
        train_val_data = split_train_val(
            X_train=X,
            y_train=y,
            categorical_features=categorical_indicator,
            task_type="regression",
            val_size=0.2,
            random_state=0,
        )

        # Generate info
        info = generate_info(
            categorical_features=categorical_indicator, task_type="regression"
        )
        self.info = info

        # Retrieve the appropriate method based on model_type
        method = get_method(self.model_type)(self, is_regression=True)

        # Fit the model using the training data
        with (
            patch("torch.save", lambda x, y: None),
            patch("torch.load", lambda x: {"params": None}),
            patch("pickle.dump", lambda x, y: None),
        ):
            with SuppressPrint():
                time_cost = method.fit(
                    train_val_data, info, train=True, config=self.config
                )

        self.method = method

        return self

    def predict(self, X):
        """
        Make predictions with the trained regression model.
        Parameters:
            X: Features (pre-split test data)
            categorical_indicator: List indicating which features are categorical
        Returns:
            Predictions as a 1D NumPy array
        """
        # Convert test data
        test_data = convert_test(
            X_test=X, categorical_features=self.categorical_indicator
        )

        # Make predictions
        with (
            patch("torch.load", lambda x: {"params": None}),
            patch("pickle.load", lambda x: self.method.model),
            patch.object(self.method, "metric", lambda x, y, z: (tuple(), tuple())),
        ):
            if hasattr(self.method.model, "load_state_dict"):
                with patch.object(self.method.model, "load_state_dict", lambda x: x):
                    prediction = self.make_prediction(test_data)
            else:
                prediction = self.make_prediction(test_data)

        # TALENT predict already denormalizes mean_std y. Do not scale again.
        return prediction.flatten()

    def make_prediction(self, test_data):
        # Classical TALENT methods return 3 values, deep methods 4. ExtraTrees is
        # classical but may be missing from this module's classical_models copy.
        return self.method.predict(
            test_data, self.info, model_name=self.evaluate_option
        )[-1]


if __name__ == "__main__":
    results_list, time_list = [], []
    for model in [
        "modernNCA",  # ICLR 2025
        "xgboost",  # KDD 2014
    ]:
        e = DeepRegressor(model_type=model)
        dataset = openml.datasets.get_dataset(
            549,
            download_data=True,
            download_qualities=True,
            download_features_meta_data=True,
        )
        qualities = dataset.qualities
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute, dataset_format="dataframe"
        )
        X, y = np.array(X), np.array(y)
        X_train_pre, X_test, y_train_pre, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0, shuffle=True
        )
        e.fit(X_train_pre, y_train_pre, categorical_indicator)
        print(e.predict(X_test))
