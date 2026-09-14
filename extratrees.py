"""ExtraTrees on TALENT RandomForest configs."""

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from TALENT.model.classical_methods.randomforest import RandomForestMethod
from TALENT.model.method_registry import (
    METHOD_REGISTRY,
    Architecture,
    Hardware,
    MethodSpec,
    OutputType,
)


class ExtraTreesMethod(RandomForestMethod):
    def construct_model(self, model_config=None):
        if model_config is None:
            model_config = self.args.config["model"]
        cls = ExtraTreesRegressor if self.is_regression else ExtraTreesClassifier
        self.model = cls(**model_config, random_state=self.args.seed)


def talent_regressor(model_type=None, **kwargs):
    """DeepRegressor factory. ExtraTrees reuses RandomForest JSON, then remaps."""
    from experiment.scikit_talent.talent_regressor import DeepRegressor

    if model_type == "ExtraTrees":
        return extra_trees_from_rf(DeepRegressor(model_type="RandomForest", **kwargs))
    return DeepRegressor(model_type=model_type, **kwargs)


def extra_trees_from_rf(regr):
    """``regr`` must be a DeepRegressor already inited as RandomForest."""
    from experiment.scikit_talent.talent_classifier import classical_models

    if "ExtraTrees" not in classical_models:
        classical_models.append("ExtraTrees")
    if "ExtraTrees" not in METHOD_REGISTRY:
        METHOD_REGISTRY["ExtraTrees"] = MethodSpec(
            name="ExtraTrees",
            module="experiment.scikit_talent.extratrees",
            class_name="ExtraTreesMethod",
            architecture=Architecture.CLASSICAL,
            hardware=Hardware.CPU,
            output_type=OutputType.PROBABILITIES,
        )
    regr.model_type = "ExtraTrees"
    regr.opt_space = {"ExtraTrees": regr.opt_space["RandomForest"]}
    regr.default_para = {"ExtraTrees": regr.default_para["RandomForest"]}
    return regr
