import os
import torch
import csv
import pdb
import pickle
import numpy as np
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.utils.data.dataset import random_split
from torch_geometric.nn import GraphConv, global_mean_pool
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

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


# Create the loss function
loss_fn = torch.nn.MSELoss()

torch.manual_seed(2024)

# Define the number of features for the nodes
num_node_features = 4


def run_gcn_mse(hp):
    # Retrieve job_id from hyperparameters, if available
    job_id = hp.get("job_id", "unknown")

    # # Mix up the seed for different evals
    # if job_id == 'unknown':
    #     seed = 2024
    # else:
    #     seed = int(job_id) + 2024

    # Set always the same seed this time
    seed = 2024

    torch.manual_seed(seed)
    random.seed(seed)

    # Load train_and_val_graph_list from pickle file
    with open("../Data/train_and_val_graph_list.pkl", "rb") as f:
        train_and_val_graph_list = pickle.load(f)

    # # Generate shuffled indices
    # indices = list(range(len(train_and_val_graph_list)))
    # random.shuffle(indices)

    # # Split the indices into training and validation sets
    # num_train = int(len(indices) * 0.8)
    # train_indices = indices[:num_train]
    # val_indices = indices[num_train:]

    # # Create a list of 0 and 1 values indicating whether each index is in train_data or val_data
    # train_val_split = [1 if i in train_indices else 0 for i in range(len(train_and_val_graph_list))]

    # Create directory for saving the split information
    split_dir = "../train_val_splits/"
    # os.makedirs(split_dir, exist_ok=True)

    # # Save the train_val_split list using pickle
    # split_file_path = os.path.join(split_dir, f'train_val_split_{job_id}.pkl')
    # with open(split_file_path, 'wb') as f:
    #     pickle.dump(train_val_split, f)
    split_file_path = os.path.join(split_dir, f"random_split.pkl")
    with open(split_file_path, "rb") as f:
        train_val_split = pickle.load(f)

    # Create training and validation datasets using the indices
    train_data = [train_and_val_graph_list[i] for i in train_val_split["train"]]
    val_data = [train_and_val_graph_list[i] for i in train_val_split["val"]]

    # Create DataLoaders for each dataset
    train_loader = DataLoader(train_data, batch_size=hp["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=hp["batch_size"], shuffle=False)

    # Create the model and move it to the device
    model = GCNNet_mse(num_node_features, hp).to(device)

    # Create a list to store validation losses
    val_losses = []

    # Create the optimizer
    optimizer = optim.Adam(model.parameters(), lr=hp["lr"])

    # Create the scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        patience=hp["patience"],
        factor=hp["scheduler_factor"],
        min_lr=1e-6,
    )

    train_losses = []

    # Training loop
    for epoch in range(hp["epochs"]):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = loss_fn(output, batch.y.float())
            train_loss = loss.detach().cpu().numpy()
            train_loss = float(train_loss)
            # mu, std = model(batch)
            # loss = loss_fn(mu, batch.y, std)
            loss.backward()
            optimizer.step()
            train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                # mu, std = model(batch)
                # val_loss += loss_fn(mu, batch.y, std).item()
                val_loss += loss_fn(output, batch.y.float()).item()
                val_loss = float(val_loss)
            val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} of job {job_id}, Validation Loss: {val_loss:.7f}")

        # Step the scheduler
        scheduler.step(val_loss)

        # Append the epoch and validation loss to the list
        val_losses.append([epoch + 1, val_loss])

    # Save the model
    model_save_dir = "../models/gcn/"
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f"model_{job_id}.pth")
    torch.save(model.state_dict(), model_save_path)

    # Return the negative final validation loss as the metric for hyperparameter tuning
    return {"objective": -val_loss, "metadata": {"val_loss": val_loss}}

    # return model


if __name__ == "__main__":
    hp = {
        "job_id": "testing_gcn",
        "batch_size": 32,
        "patience": 10,
        "scheduler_factor": 0.8,
        "hidden_dim1": 128,
        "hidden_dim2": 128,
        "hidden_dim3": 128,
        "hidden_dim4": 128,
        "hidden_dim5": 128,
        "hidden_dim6": 128,
        "num_layers1": 1,
        "num_layers2": 1,
        "num_layers3": 1,
        "num_layers4": 1,
        "num_layers5": 1,
        "num_layers6": 1,
        "lr": 0.00005,
        "epochs": 200,
        "use_batch_norm": True,
    }

    run_gcn_mse(hp)
