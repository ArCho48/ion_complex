import os
import sys
import csv
import pdb
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

from deephyper.ensemble import EnsemblePredictor
from deephyper.ensemble.loss import SquaredError
from deephyper.ensemble.aggregator import MeanAggregator
from deephyper.ensemble.selector import GreedySelector, TopKSelector

from Training_svm_mse import dense_and_concat


def create_svm_ensemble(
    model_checkpoint_dir, val_data, val_labels, ensemble_selector: str = "topk", k=100
):
    predictors = []
    for file_name in os.listdir(model_checkpoint_dir):
        if not file_name.endswith(".pth"):
            continue

        model_path = os.path.join(model_checkpoint_dir, file_name)

        with open(model_path, "rb") as f:
            predictors.append(pickle.load(f))

    # Build an ensemble
    ensemble = EnsemblePredictor(
        predictors=predictors,
        aggregator=MeanAggregator(with_uncertainty=True),
    )
    y_predictors = ensemble.predictions_from_predictors(
        val_data, predictors=ensemble.predictors
    )

    # Use TopK or Greedy/Caruana
    if ensemble_selector == "topk":
        selector = TopKSelector(
            loss_func=SquaredError(),
            k=k,
        )
    elif ensemble_selector == "greedy":
        selector = GreedySelector(
            loss_func=SquaredError(),
            aggregator=MeanAggregator(),
            k=k,
            k_init=10,
            eps_tol=1e-5,
        )
    else:
        raise ValueError(f"Unknown ensemble_selector: {ensemble_selector}")

    selected_predictors_indexes, selected_predictors_weights = selector.select(
        val_labels, y_predictors
    )

    return (ensemble, selected_predictors_indexes, selected_predictors_weights)


if __name__ == "__main__":
    seed = 2024

    random.seed(seed)

    # Load train_and_val_list from pickle file
    with open("../Data/train_and_val_list.pkl", "rb") as f:
        train_and_val_list = pickle.load(f)

    # Create directory for saving the split information
    split_dir = "../train_val_splits/"
    split_file_path = os.path.join(split_dir, f"random_split.pkl")
    with open(split_file_path, "rb") as f:
        train_val_split = pickle.load(f)

    # Dense and concat
    label_list = train_and_val_list["labels"]
    concat = False
    train_and_val_list = dense_and_concat(train_and_val_list, concat=concat)
    input_dim = train_and_val_list.shape[1]

    # Create training and validation datasets using the indices
    val_data, val_labels = (
        train_and_val_list[train_val_split["val"]],
        label_list[train_val_split["val"]],
    )

    # Load model
    model_checkpoint_dir = "../models/svm/"

    k = 100

    (
        ensemble,
        selected_predictors_indexes,
        selected_predictors_weights,
    ) = create_svm_ensemble(model_checkpoint_dir, val_data, val_labels, sys.argv[1], k)

    ensemble.predictors = [ensemble.predictors[i] for i in selected_predictors_indexes]
    ensemble.weights = selected_predictors_weights

    # Load test_graph_list from pickle file
    with open("../Data/test_list.pkl", "rb") as f:
        test_list = pickle.load(f)

    test_data, test_labels = (
        np.asarray(test_list["feats"].todense()),
        test_list["labels"],
    )

    test_pred = ensemble.predict(test_data)
    test_mse = mean_squared_error(test_labels, test_pred["loc"])
    print(test_mse)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot with error bars
    ax.errorbar(test_labels, test_pred["loc"], yerr=test_pred["uncertainty"], fmt='o', alpha=0.7, label='Model Predictions', ecolor='gray', elinewidth=3, capsize=0)
    ax.plot(
        [test_labels.min(), test_labels.max()],
        [test_labels.min(), test_labels.max()], 'k--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('$\Delta G_{ox}$ (eV) - Actual Values',fontsize=15)
    ax.set_ylabel('$\Delta G_{ox}$ (eV) - Predicted Values',fontsize=15)
    ax.set_title('SVM Ensemble ('+ sys.argv[1] +') Predictions',fontsize=17)
    ax.legend(fontsize=13)
    ax.grid(True)
    plt.tight_layout()
    name = 'fig1' if sys.argv[1] == 'greedy' else 'fig2'
    plt.savefig("../results/svm/"+name+".png")
