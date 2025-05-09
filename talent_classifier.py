import json
import os
import os.path as osp
import sys
from pprint import pprint
from unittest.mock import patch

os.environ["TQDM_DISABLE"] = "1"

from contextlib import contextmanager

import numpy as np
import openml
import torch
from data_loader import (
    convert_test,
    generate_info,
    split_train_val,
)
from TALENT.model.utils import get_method, mkdir, set_gpu, set_seeds
from sklearn.base import BaseEstimator, ClassifierMixin  # or RegressorMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout  # Save original stdout
        sys.stdout = open(os.devnull, "w")  # Redirect stdout to null

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()  # Close the null output stream
        sys.stdout = self._original_stdout  # Restore stdout


@contextmanager
def patch_if_exist(obj, attr, new):
    """Patch an attribute if it exists on the given object."""
    if hasattr(obj, attr):
        with patch.object(obj, attr, new) as p:
            yield p
    else:
        yield None


classical_models = [
    "LogReg",
    "NCM",
    "RandomForest",
    "xgboost",
    "catboost",
    "lightgbm",
    "svm",
    "knn",
    "NaiveBayes",
    "dummy",
    "LinearRegression",
]


class DeepClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        dataset=None,
        model_type=None,
        max_epoch=None,
        batch_size=None,
        normalization=None,
        num_nan_policy=None,
        cat_nan_policy=None,
        cat_policy=None,
        num_policy=None,
        n_bins=None,
        cat_min_frequency=None,
        n_trials=None,
        seed_num=None,
        workers=None,
        gpu=None,
        tune=False,
        retune=False,
        evaluate_option=None,
        dataset_path=None,
        model_path=None,
        talent_path="LAMDA-TALENT/LAMDA_TALENT",
    ):
        """
        Initialize the DeepModelEstimator with given parameters.
        Parameters not provided will be loaded from configuration files.
        """
        self.talent_path = talent_path
        # Load default arguments from JSON config
        if model_type in classical_models:
            path = f"{self.talent_path}/configs/classical_configs.json"
        else:
            path = f"{self.talent_path}/configs/deep_configs.json"
        with open(path, "r") as file:
            default_args = json.load(file)

        # Assign parameters, using defaults where necessary
        self.dataset = dataset if dataset is not None else default_args.get("dataset")
        self.model_type = (
            model_type if model_type is not None else default_args.get("model_type")
        )
        self.max_epoch = (
            max_epoch if max_epoch is not None else default_args.get("max_epoch")
        )
        self.batch_size = (
            batch_size if batch_size is not None else default_args.get("batch_size")
        )
        self.normalization = (
            normalization
            if normalization is not None
            else default_args.get("normalization")
        )
        self.num_nan_policy = (
            num_nan_policy
            if num_nan_policy is not None
            else default_args.get("num_nan_policy")
        )
        self.cat_nan_policy = (
            cat_nan_policy
            if cat_nan_policy is not None
            else default_args.get("cat_nan_policy")
        )
        self.cat_policy = (
            cat_policy if cat_policy is not None else default_args.get("cat_policy")
        )
        self.num_policy = (
            num_policy if num_policy is not None else default_args.get("num_policy")
        )
        if self.model_type in [
            "tabr",
            "modernNCA",
            "mlp_plr",
        ]:
            self.cat_policy = "tabr_ohe"
        if self.model_type in [
            "autoint",
            "dcn2",
            "ftt",
            "grownet",
            "ptarl",
            "saint",
            "snn",
            "tabtransformer",
            "realmlp",
            "trompt",
            "amformer",
            "grande",
            "bishop",
            "tabpfn",
        ]:
            self.cat_policy = "indices"
        if self.model_type in ["tabpfn"]:
            self.normalization = "none"
        self.n_bins = n_bins if n_bins is not None else default_args.get("n_bins")
        self.cat_min_frequency = (
            cat_min_frequency
            if cat_min_frequency is not None
            else default_args.get("cat_min_frequency")
        )
        self.n_trials = (
            n_trials if n_trials is not None else default_args.get("n_trials")
        )
        self.seed_num = (
            seed_num if seed_num is not None else default_args.get("seed_num")
        )
        self.workers = workers if workers is not None else default_args.get("workers")
        self.gpu = gpu if gpu is not None else default_args.get("gpu")
        self.tune = tune if tune is not None else default_args.get("tune")
        self.retune = retune if retune is not None else default_args.get("retune")
        self.evaluate_option = (
            evaluate_option
            if evaluate_option is not None
            else default_args.get("evaluate_option")
        )
        self.dataset_path = (
            dataset_path
            if dataset_path is not None
            else default_args.get("dataset_path")
        )
        self.model_path = (
            model_path if model_path is not None else default_args.get("model_path")
        )

        # Set up GPU
        set_gpu(self.gpu)

        # Create save paths
        save_path1 = "-".join([self.dataset, self.model_type])
        save_path2 = f"Epoch{self.max_epoch}BZ{self.batch_size}"
        save_path2 += f"-Norm-{self.normalization}"
        save_path2 += f"-Nan-{self.num_nan_policy}-{self.cat_nan_policy}"
        save_path2 += f"-Cat-{self.cat_policy}"

        if self.cat_min_frequency > 0.0:
            save_path2 += f"-CatFreq-{self.cat_min_frequency}"
        if self.tune:
            save_path1 += "-Tune"

        save_path = osp.join(save_path1, save_path2)
        self.save_path = osp.join(self.model_path, save_path)
        mkdir(self.save_path)

        # Load additional configurations
        config_default_path = os.path.join(
            talent_path,
            "configs",
            "default",
            f"{self.model_type}.json",
        )
        config_opt_path = os.path.join(
            talent_path,
            "configs",
            "opt_space",
            f"{self.model_type}.json",
        )
        with open(config_default_path, "r") as file:
            default_para = json.load(file)

        with open(config_opt_path, "r") as file:
            opt_space = json.load(file)

        self.config = default_para[self.model_type]
        if model_type in classical_models:
            self.config["fit"]["n_bins"] = self.n_bins
        else:
            self.config["training"]["n_bins"] = self.n_bins

        # Set seeds
        self.seed = 0  # You might want to use self.seed_num instead
        set_seeds(self.seed)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # For debugging: print all parameters
        pprint(self.get_params())

        # Store additional configurations
        self.default_para = default_para
        self.opt_space = opt_space

    def fit(self, X, y, categorical_indicator):
        """
        Fit the deep learning model.
        Parameters:
            X: Features
            y: Target variable
        Returns:
            self
        """
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        # Now, use the two functions to further split train into train and val, and convert test
        self.categorical_indicator = categorical_indicator
        train_data = split_train_val(
            X_train=X,
            y_train=y,
            categorical_features=categorical_indicator,
            task_type="binclass",
            val_size=0.2,
            random_state=0,
        )
        # Generate info
        info = generate_info(
            categorical_features=categorical_indicator, task_type="binclass"
        )
        if np.unique(y).shape[0] > 2:
            info["task_type"] = "multiclass"
        else:
            info["task_type"] = "binclass"
        self.info = info
        method = get_method(self.model_type)(self, info["task_type"] == "regressor")

        with (
            patch("torch.save", lambda x, y: None),
            patch("torch.load", lambda x: {"params": None}),
            patch("pickle.dump", lambda x, y: None),
        ):
            with SuppressPrint():
                time_cost = method.fit(train_data, info, train=True, config=self.config)
        self.method = method
        self.default_y = np.unique(y)

    def predict(self, X):
        """
        Make predictions with the trained model.
        Parameters:
            X: Features
        Returns:
            Predictions
        """
        test_data = convert_test(
            X_test=X,
            categorical_features=self.categorical_indicator,
            default_y=self.default_y,
        )

        with (
            patch("torch.load", lambda x: {"params": None}),
            patch("pickle.load", lambda x: self.method.model),
            patch.object(self.method, "metric", lambda x, y, z: (tuple(), tuple())),
        ):
            with patch_if_exist(self.method.model, "load_state_dict", lambda x: x):
                if self.model_type in classical_models:
                    _, _, predict_logits = self.method.predict(
                        test_data, self.info, model_name=self.evaluate_option
                    )
                else:
                    _, _, _, predict_logits = self.method.predict(
                        test_data, self.info, model_name=self.evaluate_option
                    )
        if len(np.shape(predict_logits)) == 2:
            # probabilistic output
            prediction = np.argmax(predict_logits, axis=1)
        else:
            prediction = predict_logits
        prediction = self.method.label_encoder.inverse_transform(prediction)
        prediction = self.label_encoder.inverse_transform(prediction)

        return prediction


if __name__ == "__main__":
    torch.set_num_threads(1)
    results_list, time_list = [], []
    for model in [
        "modernNCA",  # ICLR 2025
        "xgboost",  # KDD 2014
    ]:
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
        X_train_pre, X_test, y_train_pre, y_test = train_test_split(
            X, y, train_size=100, random_state=0, shuffle=True, stratify=y
        )
        e.fit(X_train_pre, y_train_pre, categorical_indicator)
        print(model, balanced_accuracy_score(y_test, e.predict(X_test)))
