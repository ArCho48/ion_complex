import os
import sys
import csv
import pdb
import tqdm
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader

from deephyper.analysis.hpo import parameters_from_row
from deephyper.ensemble.aggregator import MeanAggregator
from deephyper.ensemble.loss import SquaredError
from deephyper.predictor.torch import TorchPredictor
from deephyper.ensemble.selector import GreedySelector, TopKSelector
from deephyper.ensemble import EnsemblePredictor

del EnsemblePredictor.__del__

from Training_mlp_mse import MLP_mse, myDataset, dense_and_concat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLPTorchPredictor(TorchPredictor):
    """Represents a frozen torch model that can only predict."""

    def __init__(self, module: torch.nn.Module):
        if not isinstance(module, torch.nn.Module):
            raise ValueError(
                f"The given module is of type {type(module)} when it should be of type "
                f"torch.nn.Module!"
            )
        self.module = module

    def pre_process_inputs(self, x):
        return x

    def post_process_predictions(self, y):
        y = y.detach().cpu().numpy()
        return y

    def predict(self, X):
        X = self.pre_process_inputs(X)
        training = self.module.training
        if training:
            self.module.eval()

        if hasattr(self.module, "predict_proba"):
            y = self.module.predict_proba(X)
        else:
            y = self.module(X)

        self.module.train(training)
        y = self.post_process_predictions(y)
        return y


def create_mlp_ensemble(
    hpo_results,
    model_checkpoint_dir,
    input_dim,
    val_data,
    ensemble_selector: str = "topk",
    k=100,
):
    y_predictors = []
    valid_y = []
    job_id_predictors = []

    for file_name in os.listdir(model_checkpoint_dir):
        if not file_name.endswith(".pth"):
            continue

        weights_path = os.path.join(model_checkpoint_dir, file_name)
        job_id = int(file_name[6:-3].split(".")[0])

        row = hpo_results[hpo_results["job_id"] == job_id]
        if len(row) == 0:
            continue
        assert len(row) == 1
        row = row.iloc[0]
        parameters = parameters_from_row(row)

        # batch size
        val_loader = DataLoader(
            val_data, batch_size=parameters["batch_size"], shuffle=False
        )

        # Create the model and move it to the device
        model = MLP_mse(input_dim, parameters).to(device)

        try:
            model.load_state_dict(
                torch.load(weights_path, weights_only=True, map_location=device)
            )
        except RuntimeError:
            continue

        model.eval()
        with torch.no_grad():
            batch_y_pred = []
            batch_y_valid = []
            for batch in val_loader:
                batch[0] = batch[0].to(device)
                batch_y_pred.append(model(batch[0]).detach().cpu())
                batch_y_valid.append(batch[1].float())

        y_predictors.append(np.concatenate(batch_y_pred))
        valid_y.append(np.concatenate(batch_y_valid))
        job_id_predictors.append(job_id)

    y_predictors = np.array(y_predictors)
    valid_y = np.array(valid_y)

    ## Use TopK or Greedy/Caruana
    if ensemble_selector == "topk":
        selector = TopKSelector(
            loss_func=SquaredError(),
            k=k,
        )
    else:
        selector = GreedySelector(
            loss_func=SquaredError(),
            aggregator=MeanAggregator(),
            k=k,
            max_it=200,
            k_init=10,
            early_stopping=True,
            with_replacement=True,
            bagging=True,
            verbose=False,
        )

    selected_predictors_indexes, selected_predictors_weights = selector.select(
        valid_y,
        y_predictors,
    )

    selected_predictors_job_ids = np.array(job_id_predictors)[
        selected_predictors_indexes
    ]

    return (selected_predictors_job_ids, selected_predictors_weights)


if __name__ == "__main__":
    # Set always the same seed
    seed = 2024

    torch.manual_seed(seed)
    random.seed(seed)

    # Load train_and_val_graph_list from pickle file
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
    # train_data = myDataset( train_and_val_list[train_val_split['train']], label_list[train_val_split['train']] )
    val_data = myDataset(
        train_and_val_list[train_val_split["val"]], label_list[train_val_split["val"]]
    )

    # Load results and model
    hpo_results = pd.read_csv("../results/mlp/results.csv")
    model_checkpoint_dir = "../models/mlp/"

    k = 100

    selected_predictors_job_ids, selected_predictors_weights = create_mlp_ensemble(
        hpo_results, model_checkpoint_dir, input_dim, val_data, sys.argv[1], k
    )

    # Load test_graph_list from pickle file
    with open("../Data/test_list.pkl", "rb") as f:
        test_list = pickle.load(f)

    test_data = myDataset(np.asarray(test_list["feats"].todense()), test_list["labels"])

    predictors = []
    model_checkpoint_dir = "../models/mlp/"

    for index, job_id in enumerate(selected_predictors_job_ids):
        file_name = f"model_{job_id}.pth"

        weights_path = os.path.join(model_checkpoint_dir, file_name)

        row = hpo_results[hpo_results["job_id"] == job_id].iloc[0]
        parameters = parameters_from_row(row)

        # batch size
        test_loader = DataLoader(
            test_data, batch_size=parameters["batch_size"], shuffle=False
        )

        # Create the model and move it to the device
        model = MLP_mse(input_dim, parameters).to(device)

        model.load_state_dict(
            torch.load(weights_path, weights_only=True, map_location=device)
        )
        model.eval()

        predictors.append(MLPTorchPredictor(model))

    ensemble = EnsemblePredictor(
        predictors=predictors,
        aggregator=MeanAggregator(with_uncertainty=True),
        weights=selected_predictors_weights,
    )

    test_error = []
    y_pred_mean_store = []
    y_pred_std_store = []
    test_y = []
    for batch in test_loader:
        batch[0] = batch[0].to(device)
        y_pred = ensemble.predict(batch[0])
        y_pred_mean_store.append(y_pred["loc"])
        y_pred_std_store.append(np.sqrt(y_pred["uncertainty"]))
        test_y.append(batch[1])
        # pdb.set_trace()
        test_error.append(y_pred["loc"] - (batch[1].cpu().numpy()))
    test_error = np.concatenate(test_error)
    test_mse = np.mean(np.square(test_error))

    print(test_mse)

    y_pred_mean_store = np.concatenate(y_pred_mean_store)
    y_pred_std_store = np.concatenate(y_pred_std_store)
    test_y = np.concatenate(test_y)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot with error bars
    ax.errorbar(test_y, y_pred_mean_store, yerr=y_pred_std_store, fmt='o', alpha=0.7, label='Model Predictions', ecolor='gray', elinewidth=3, capsize=0)
    ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('$\Delta G_{ox}$ (eV) - Actual Values',fontsize=15)
    ax.set_ylabel('$\Delta G_{ox}$ (eV) - Predicted Values',fontsize=15)
    ax.set_title('MLP Ensemble ('+ sys.argv[1] +') Predictions',fontsize=17)
    ax.legend(fontsize=13)
    ax.grid(True)
    plt.tight_layout()
    name = 'fig1' if sys.argv[1] == 'greedy' else 'fig2'
    plt.savefig("../results/mlp/"+name+".png")
