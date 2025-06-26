# main_search.py
from deephyper.hpo import HpProblem, CBO
from deephyper.evaluator import Evaluator
from Training_gcn_mse import run_gcn_mse
from Training_svm import run_svm
from Training_rf_mse import run_rf
from Training_mlp_mse import run_mlp_mse
import sys

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

if sys.argv[1] == "gcn":
    run_function = run_gcn_mse
elif sys.argv[1] == "mlp":
    run_function = run_mlp_mse
elif sys.argv[1] == "svm":
    run_function = run_svm
elif sys.argv[1] == "rf":
    run_function = run_rf

# Ray initialization
# If the following line is used then you need to set the parameters for cpus/gpus yourself.
# ray.init()

# Problem definition
hp = HpProblem()
if sys.argv[1] == "gcn":
    hp.add_hyperparameter([True, False], "use_batch_norm", default_value=True)
    hp.add_hyperparameter((16, 128), "batch_size", default_value=32)
    hp.add_hyperparameter((3, 10), "patience", default_value=10)
    hp.add_hyperparameter((0.1, 0.9), "scheduler_factor", default_value=0.8)
    hp.add_hyperparameter((16, 256), "hidden_dim1", default_value=128)
    hp.add_hyperparameter((16, 256), "hidden_dim2", default_value=128)
    hp.add_hyperparameter((16, 256), "hidden_dim3", default_value=128)
    hp.add_hyperparameter((16, 256), "hidden_dim4", default_value=128)
    hp.add_hyperparameter((16, 256), "hidden_dim5", default_value=128)
    hp.add_hyperparameter((16, 256), "hidden_dim6", default_value=128)
    hp.add_hyperparameter((1, 3), "num_layers1", default_value=1)
    hp.add_hyperparameter((1, 3), "num_layers2", default_value=1)
    hp.add_hyperparameter((1, 3), "num_layers3", default_value=1)
    hp.add_hyperparameter((1, 3), "num_layers4", default_value=1)
    hp.add_hyperparameter((1, 3), "num_layers5", default_value=1)
    hp.add_hyperparameter((1, 3), "num_layers6", default_value=1)
    hp.add_hyperparameter((1e-5, 1e-3, "log-uniform"), "lr", default_value=0.00005)
    hp.add_hyperparameter([200], "epochs")
elif sys.argv[1] == "mlp":
    hp.add_hyperparameter([True, False], "use_batch_norm", default_value=True)
    hp.add_hyperparameter((16, 128), "batch_size", default_value=32)
    hp.add_hyperparameter((3, 10), "patience", default_value=10)
    hp.add_hyperparameter((0.1, 0.9), "scheduler_factor", default_value=0.8)
    hp.add_hyperparameter((32, 512), "hidden_dim1", default_value=64)
    hp.add_hyperparameter((32, 512), "hidden_dim2", default_value=128)
    hp.add_hyperparameter((32, 512), "hidden_dim3", default_value=256)
    hp.add_hyperparameter((32, 512), "hidden_dim4", default_value=512)
    hp.add_hyperparameter((32, 512), "hidden_dim5", default_value=128)
    hp.add_hyperparameter((32, 512), "hidden_dim6", default_value=32)
    hp.add_hyperparameter((5e-4, 1e-2, "log-uniform"), "lr", default_value=0.0005)
    hp.add_hyperparameter([200], "epochs")
elif sys.argv[1] == "svm":
    hp.add_hyperparameter((1, 100), "C", default_value=1)
    hp.add_hyperparameter(["scale", "auto"], "gamma", default_value="auto")
    hp.add_hyperparameter((1, 5), "degree", default_value=3)
elif sys.argv[1] == "rf":
    # inspired from this: https://github.com/fmohr/lcdb/blob/main/publications/2023-neurips/lcdb/workflow/sklearn/_trees_ensemble.py
    from ConfigSpace import (
        Constant,
        Categorical,
        ConfigurationSpace,
        Float,
        Integer,
        EqualsCondition,
    )

    config_space = ConfigurationSpace(
        name="sklearn.TreesEnsembleWorkflow",
        space={
            "n_estimators": Constant("n_estimators", value=2048),
            "criterion": Categorical(
                "criterion",
                items=["squared_error", "absolute_error", "friedman_mse"],
                default="squared_error",
            ),
            "max_depth": Integer("max_depth", bounds=(0, 100), default=0),
            "min_samples_split": Integer(
                "min_samples_split", bounds=(2, 50), default=2
            ),
            "min_samples_leaf": Integer("min_samples_leaf", bounds=(1, 25), default=1),
            "max_features": Categorical(
                "max_features", items=["all", "sqrt", "log2"], default="sqrt"
            ),
            "min_impurity_decrease": Float(
                "min_impurity_decrease", bounds=(0.0, 1.0), default=0.0
            ),
            "bootstrap": Categorical("bootstrap", items=[True, False], default=True),
            "max_samples": Float(
                "max_samples", bounds=(1e-3, 1.0), default=1.0
            ),  # cannot be 0
            "splitter": Categorical(
                "splitter", items=["random", "best"], default="best"
            ),
        },
    )

    config_space.add_condition(
        EqualsCondition(config_space["max_samples"], config_space["bootstrap"], True)
    )
    hp = HpProblem(config_space)

if __name__ == "__main__":
    # Evaluator creation
    print(f"Creation of the Evaluator... {hp.default_configuration}")
    evaluator = Evaluator.create(
        run_function,
        method="ray",
        method_kwargs={
            "num_cpus": 8,
            "num_gpus": 0,
            "num_cpus_per_task": 1,
            "num_gpus_per_task": 0,
        },
    )
    print(f"Creation of the Evaluator done with {evaluator.num_workers} worker(s)")

    # Search creation
    print("Creation of the search instance...")
    search = CBO(
        hp,
        evaluator,
        initial_points=[hp.default_configuration],
        acq_optimizer="mixedga",
        acq_optimizer_freq=1,
        log_dir="../results/" + sys.argv[1],
    )

    results = search.search(max_evals=1200)

    print(results)
