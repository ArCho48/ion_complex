import os
import torch
import csv
import pdb
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.data.dataset import random_split
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP_mse(torch.nn.Module):
    def __init__(self, input_dim, hp):
        super(MLP_mse, self).__init__()

        self.use_batch_norm = hp["use_batch_norm"]
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList() if self.use_batch_norm else None

        hidden_dims = [hp[f"hidden_dim{i+1}"] for i in range(6)]

        # Define the input layer
        self.convs.append(torch.nn.Linear(input_dim, hidden_dims[0]))
        if self.use_batch_norm:
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dims[0]))

        # Define the hidden layers
        for i in range(len(hidden_dims) - 1):
            self.convs.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if self.use_batch_norm:
                self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dims[i + 1]))

        # Define the output layer
        self.fc_mu = torch.nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        # Fully connected layers
        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = torch.nn.functional.relu(x)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)

        # Output layer
        x = self.fc_mu(x).squeeze(-1)

        return x


class myDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        return feature, label


# Create the loss function
loss_fn = torch.nn.MSELoss()

torch.manual_seed(2024)


def dense_and_concat(x, concat=False):
    if not concat:
        return np.asarray(x["feats"].todense())
    else:
        return np.asarray(
            np.concatenate((x["feats"].todense(), x["adj"].todense()), axis=1)
        )


def run_mlp_mse(hp):
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

    # Load train_and_val_list from pickle file
    with open("../Data/train_and_val_list.pkl", "rb") as f:
        train_and_val_list = pickle.load(f)

    # # Generate shuffled indices
    # indices = list(range(train_and_val_list['feats'].shape[0]))
    # random.shuffle(indices)

    # # Split the indices into training and validation sets
    # num_train = int(len(indices) * 0.8)
    # train_indices = indices[:num_train]
    # val_indices = indices[num_train:]

    # # Create a list of 0 and 1 values indicating whether each index is in train_data or val_data
    # train_val_split = [1 if i in train_indices else 0 for i in range(len(train_and_val_list))]

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

    # Dense and concat
    label_list = train_and_val_list["labels"]
    concat = False
    train_and_val_list = dense_and_concat(train_and_val_list, concat=concat)
    input_dim = train_and_val_list.shape[1]

    # Create training and validation datasets using the indices
    train_data = myDataset(
        train_and_val_list[train_val_split["train"]],
        label_list[train_val_split["train"]],
    )
    val_data = myDataset(
        train_and_val_list[train_val_split["val"]], label_list[train_val_split["val"]]
    )

    # Create DataLoaders for each dataset
    train_loader = DataLoader(train_data, batch_size=hp["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=hp["batch_size"], shuffle=False)

    # Create the model and move it to the device
    model = MLP_mse(input_dim, hp).to(device)

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
            batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            output = model(batch[0])
            loss = loss_fn(output, batch[1].float())
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
                batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
                output = model(batch[0])
                # mu, std = model(batch)
                # val_loss += loss_fn(mu, batch.y, std).item()
                val_loss += loss_fn(output, batch[1].float()).item()
                val_loss = float(val_loss)
            val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} of job {job_id}, Validation Loss: {val_loss:.7f}")

        # Step the scheduler
        scheduler.step(val_loss)

        # Append the epoch and validation loss to the list
        val_losses.append([epoch + 1, val_loss])

    # Save the model
    model_save_dir = "../models/mlp/"
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f"model_{job_id}.pth")
    torch.save(model.state_dict(), model_save_path)

    # Return the negative final validation loss as the metric for hyperparameter tuning
    return {"objective": -val_loss, "metadata": {"val_loss": val_loss}}

    # return model


if __name__ == "__main__":
    hp = {
        "job_id": "testing_mlp",
        "batch_size": 32,
        "patience": 10,
        "scheduler_factor": 0.8,
        "hidden_dim1": 64,
        "hidden_dim2": 128,
        "hidden_dim3": 256,
        "hidden_dim4": 512,
        "hidden_dim5": 128,
        "hidden_dim6": 32,
        "lr": 0.0005,
        "epochs": 200,
        "use_batch_norm": True,
    }

    run_mlp_mse(hp)
