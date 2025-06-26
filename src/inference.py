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
from sklearn.metrics import mean_squared_error

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch_geometric.loader import DataLoader as pyg_Dataloader

from deephyper.analysis.hpo import parameters_from_row

from Training_gcn_mse import GCNNet_mse
from Training_mlp_mse import MLP_mse, myDataset, dense_and_concat

method = sys.argv[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 2024
torch.manual_seed(seed)
random.seed(seed)

print_statements = ["Default Model", "Best Model"]

## Dataset
# Load split information
split_dir = "../train_val_splits/"
split_file_path = os.path.join(split_dir, f"random_split.pkl")
with open(split_file_path, "rb") as f:
    train_val_split = pickle.load(f)

if method == "gcn":
    # # Load train_and_val_graph_list from pickle file
    # with open('train_and_val_graph_list.pkl', 'rb') as f:
    #     train_and_val_graph_list = pickle.load(f)

    # # Create directory for saving the split information
    # split_dir = 'train_val_splits/'
    # split_file_path = os.path.join(split_dir, f'random_split.pkl')
    # with open(split_file_path, 'rb') as f:
    #     train_val_split = pickle.load(f)

    # # Create training and validation datasets using the indices
    # train_data = [train_and_val_graph_list[i] for i in train_val_split['train']]
    # val_data = [train_and_val_graph_list[i] for i in train_val_split['val']]

    # Load test_graph_list from pickle file
    with open("../Data/test_graph_list.pkl", "rb") as f:
        test_data = pickle.load(f)

    # Set node feature size
    num_node_features = 4
else:
    # # Load train_and_val_list from pickle file
    # with open('train_and_val_list.pkl', 'rb') as f:
    #     train_and_val_list = pickle.load(f)

    # # Dense and concat
    # label_list = train_and_val_list['labels']
    # concat = False
    # train_and_val_list = dense_and_concat(train_and_val_list,concat=concat)
    # input_dim = train_and_val_list.shape[1]

    # # Create training and validation datasets using the indices
    # train_data, train_labels = train_and_val_list[train_val_split['train']], label_list[train_val_split['train']]
    # val_data, val_labels = train_and_val_list[train_val_split['val']], label_list[train_val_split['val']]

    # if method == 'mlp':
    #     train_data = myDataset( train_data, train_labels )
    #     val_data = myDataset( val_data, val_labels )

    # Load test_graph_list from pickle file
    with open("../Data/test_list.pkl", "rb") as f:
        test_list = pickle.load(f)

    test_data, test_labels = (
        np.asarray(test_list["feats"].todense()),
        test_list["labels"],
    )
    input_dim = test_data.shape[1]
    if method == "mlp":
        test_data = myDataset(test_data, test_labels)

if method == "gcn":
    # Load results and model
    hpo_results = pd.read_csv("../results/gcn/results.csv")
    model_checkpoint_dir = "../models/gcn/"

    row_min = hpo_results["m:val_loss"].idxmin()

    print("GCN Test MSE -")
    for index, job_id in enumerate([0, row_min]):
        file_name = f"model_{job_id}.pth"

        weights_path = os.path.join(model_checkpoint_dir, file_name)

        row = hpo_results[hpo_results["job_id"] == job_id].iloc[0]
        parameters = parameters_from_row(row)

        # batch size
        test_loader = pyg_Dataloader(
            test_data, batch_size=parameters["batch_size"], shuffle=False
        )

        # Create the model and move it to the device
        model = GCNNet_mse(num_node_features, parameters).to(device)

        model.load_state_dict(
            torch.load(weights_path, weights_only=True, map_location=device)
        )
        model.eval()

        test_error_batch = []
        test_pred = []
        test_labels = []
        for batch in test_loader:
            batch = batch.to(device)
            y_pred_test = model(batch)
            test_labels.append(batch.y.detach().cpu())
            test_pred.append(y_pred_test.detach().cpu())
            test_error_batch.append((y_pred_test - batch.y.float()).detach().cpu())
        test_error_batch = np.concatenate(test_error_batch)
        test_pred = np.concatenate(test_pred)
        test_labels = np.concatenate(test_labels)

        test_mse = np.mean(np.square(test_error_batch))

        print(print_statements[index] + ":", test_mse)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            test_labels, test_pred, alpha=0.7, label='Model Predictions'
        )  # , yerr=test_pred["uncertainty"], fmt='o', capsize=5)
        ax.plot(
            [test_labels.min(), test_labels.max()],
            [test_labels.min(), test_labels.max()],
            "k--", lw=2, label='Perfect Prediction',
        )
        ax.set_xlabel('$\Delta G_{ox}$ (eV) - Actual Values',fontsize=15)
        ax.set_ylabel('$\Delta G_{ox}$ (eV) - Predicted Values',fontsize=15)
        plt.title('GCN '+print_statements[index]+' Predictions', fontsize=17)
        ax.legend(fontsize=13)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig("../results/gcn/fig" + str(index + 3) + ".png")

elif method == "mlp":
    # Load results and model
    hpo_results = pd.read_csv("../results/mlp/results.csv")
    model_checkpoint_dir = "../models/mlp/"

    row_min = hpo_results["m:val_loss"].idxmin()

    print("MLP Test MSE -")
    for index, job_id in enumerate([0, row_min]):
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

        test_pred = []
        test_labels = []
        test_error_batch = []
        for batch in test_loader:
            batch[0] = batch[0].to(device)
            y_pred_test = model(batch[0]).detach().cpu()
            test_labels.append(batch[1].float())
            test_pred.append(y_pred_test)
            test_error_batch.append(y_pred_test - batch[1].float())
        test_error_batch = np.concatenate(test_error_batch)
        test_pred = np.concatenate(test_pred)
        test_labels = np.concatenate(test_labels)

        test_mse = np.mean(np.square(test_error_batch))

        print(print_statements[index], test_mse)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            test_labels, test_pred, alpha=0.7, label='Model Predictions'
        )  # , yerr=test_pred["uncertainty"], fmt='o', capsize=5)
        ax.plot(
            [test_labels.min(), test_labels.max()],
            [test_labels.min(), test_labels.max()],
            "k--", lw=2, label='Perfect Prediction',
        )
        ax.set_xlabel('$\Delta G_{ox}$ (eV) - Actual Values',fontsize=15)
        ax.set_ylabel('$\Delta G_{ox}$ (eV) - Predicted Values',fontsize=15)
        plt.title('MLP '+print_statements[index]+' Predictions', fontsize=17)
        ax.legend(fontsize=13)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig("../results/mlp/fig" + str(index + 3) + ".png")

elif method == "rf":
    # Load results and model
    hpo_results = pd.read_csv("../results/rf/results.csv")
    model_checkpoint_dir = "../models/rf/"

    row_min = hpo_results["m:val_loss"].idxmin()

    print("RF Test MSE -")
    for index, job_id in enumerate([0, row_min]):
        file_name = f"model_{job_id}.pth"

        model_path = os.path.join(model_checkpoint_dir, file_name)

        # Create the model and move it to the device
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        test_pred = model.predict(test_data)
        test_mse = mean_squared_error(test_labels, test_pred)

        print(print_statements[index], test_mse)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            test_labels, test_pred, alpha=0.7, label='Model Predictions'
        )  # , yerr=test_pred["uncertainty"], fmt='o', capsize=5)
        ax.plot(
            [test_labels.min(), test_labels.max()],
            [test_labels.min(), test_labels.max()],
            "k--", lw=2, label='Perfect Prediction',
        )
        ax.set_xlabel('$\Delta G_{ox}$ (eV) - Actual Values',fontsize=15)
        ax.set_ylabel('$\Delta G_{ox}$ (eV) - Predicted Values',fontsize=15)
        plt.title('RF '+print_statements[index]+' Predictions', fontsize=17)
        ax.legend(fontsize=13)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig("../results/rf/fig" + str(index + 3) + ".png")

elif method == "svm":
    # Load results and model
    hpo_results = pd.read_csv("../results/svm/results.csv")
    model_checkpoint_dir = "../models/svm/"

    row_min = hpo_results["m:val_loss"].idxmin()

    print("SVM Test MSE -")
    for index, job_id in enumerate([0, row_min]):
        file_name = f"model_{job_id}.pth"

        model_path = os.path.join(model_checkpoint_dir, file_name)

        # Create the model and move it to the device
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        test_pred = model.predict(test_data)
        test_mse = mean_squared_error(test_labels, test_pred)

        print(print_statements[index], test_mse)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            test_labels, test_pred, alpha=0.7, label='Model Predictions'
        )  # , yerr=test_pred["uncertainty"], fmt='o', capsize=5)
        ax.plot(
            [test_labels.min(), test_labels.max()],
            [test_labels.min(), test_labels.max()],
            "k--", lw=2, label='Perfect Prediction',
        )
        ax.set_xlabel('$\Delta G_{ox}$ (eV) - Actual Values',fontsize=15)
        ax.set_ylabel('$\Delta G_{ox}$ (eV) - Predicted Values',fontsize=15)
        plt.title('RF '+print_statements[index]+' Predictions', fontsize=17)
        ax.legend(fontsize=13)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig("../results/svm/fig" + str(index + 3) + ".png")
