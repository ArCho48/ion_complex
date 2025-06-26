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
from torch.utils.data.dataset import random_split
from torch_geometric.nn import GraphConv, global_mean_pool
from torch_geometric.loader import DataLoader as Dataloader
from torch_geometric.explain import Explainer, GNNExplainer

from deephyper.analysis.hpo import parameters_from_row

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 2024
torch.manual_seed(seed)
random.seed(seed)

class GCNNet_mse(torch.nn.Module):
    def __init__(self, num_node_features, hp):
        super(GCNNet_mse, self).__init__()

        self.use_batch_norm = hp["use_batch_norm"]
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList() if self.use_batch_norm else None

        hidden_dims = [hp[f"hidden_dim{i+1}"] for i in range(6)]
        num_layers = [hp[f"num_layers{i+1}"] for i in range(6)]

        # Define the Graph Convolution layers
        for i in range(len(hidden_dims)):
            for j in range(num_layers[i]):
                if i == 0 and j == 0:
                    in_features = num_node_features
                elif j == 0:
                    in_features = hidden_dims[i - 1]
                else:
                    in_features = hidden_dims[i]
                out_features = hidden_dims[i]
                self.convs.append(GraphConv(in_features, out_features))
                if self.use_batch_norm:
                    self.batch_norms.append(torch.nn.BatchNorm1d(out_features))

        # Define the dense layer
        self.dense = torch.nn.Linear(hidden_dims[-1], hidden_dims[-1])

        # Define the output layer
        self.fc_mu = torch.nn.Linear(hidden_dims[-1], 1)

    def forward(self, x, edge_index):
        # Graph Convolution layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = torch.nn.functional.relu(x)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)

        # Aggregate node features into a single graph-level representation
        x = global_mean_pool(x, data.batch)

        # Dense layer
        x = self.dense(x)
        x = torch.nn.functional.relu(x)

        # Output layer
        mu = self.fc_mu(x).squeeze(-1)

        return mu


# Load test_graph_list from pickle file
with open("../Data/test_graph_list.pkl", "rb") as f:
    test_data = pickle.load(f)

# Set node feature size
num_node_features = 4

 # Load results and model
hpo_results = pd.read_csv("../results/gcn/results.csv")
model_checkpoint_dir = "../models/gcn/"

row_min = hpo_results["m:val_loss"].idxmin()
file_name = f"model_{row_min}.pth"
weights_path = os.path.join(model_checkpoint_dir, file_name)

row = hpo_results[hpo_results["job_id"] == row_min].iloc[0]
parameters = parameters_from_row(row)

# # batch size
# test_loader = Dataloader(
#     test_data, batch_size=parameters["batch_size"], shuffle=False
# )

# Create the model and move it to the device
model = GCNNet_mse(num_node_features, parameters).to(device)

model.load_state_dict(
    torch.load(weights_path, weights_only=True, map_location=device)
)

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type="model",
    node_mask_type='attributes',
    edge_mask_type="object",
    model_config=dict(
        mode="regression",
        task_level="graph",
        return_type="raw",
    ),
    # Include only the top 10 most important edges:
    threshold_config=dict(threshold_type="topk", value=10),
)

model_save_dir = "../results/explanability_plots/"
os.makedirs(model_save_dir, exist_ok=True)
for indx, data in enumerate(test_data):
    explanation = explainer(data.x, data.edge_index)
    # print(f'Generated explanations in {explanation.available_explanations}')

    path = os.path.join(model_save_dir,'feature_importance_'+str(indx)+'.png')
    explanation.visualize_feature_importance(path, top_k=10)
    # print(f"Feature importance plot has been saved to '{path}'")

    path = os.path.join(model_save_dir,'subgraph_'+str(indx)+'.png')
    explanation.visualize_graph(path)
    # print(f"Subgraph visualization plot has been saved to '{path}'")

