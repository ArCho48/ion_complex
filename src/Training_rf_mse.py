import os
import torch
import csv
import pdb
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
import random
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)


def dense_and_concat(x, concat=False):
    if not concat:
        return np.asarray(x["feats"].todense())
    else:
        return np.asarray(
            np.concatenate((x["feats"].todense(), x["adj"].todense()), axis=1)
        )


def fit_and_score_rf(
    train_data,
    train_labels,
    val_data,
    val_labels,
    n_estimators=2048,
    criterion="squared_error",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="sqrt",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    ccp_alpha=0.0,
    max_samples=None,
    splitter="best",
    random_state=None,
):
    max_depth = max_depth if (max_depth is not None and max_depth > 0) else None
    max_samples = max_samples if bootstrap else None
    max_features = 1.0 if max_features == "all" else max_features

    if splitter == "best":
        learner = RandomForestRegressor(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            random_state=random_state,
        )
    elif splitter == "random":
        learner = ExtraTreesRegressor(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            random_state=random_state,
        )
    else:
        raise ValueError(
            f"The splitter is '{splitter}' when it should be in ['random', 'best']."
        )

    learner.fit(train_data, train_labels)
    output = learner.predict(val_data)
    val_loss = mean_squared_error(val_labels, output)
    val_losses = [val_loss]
    train_losses = [mean_squared_error(train_labels, learner.predict(train_data))]

    return learner, val_loss, train_losses, val_losses


def run_rf(hp):
    # Retrieve job_id from hyperparameters, if available
    job_id = hp.get("job_id", "unknown")

    # # Mix up the seed for different evals
    # if job_id == 'unknown':
    #     seed = 2024
    # else:
    #     seed = int(job_id) + 2024

    # Set always the same seed this time
    seed = 2024

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
    train_data, train_labels = (
        train_and_val_list[train_val_split["train"]],
        label_list[train_val_split["train"]],
    )
    val_data, val_labels = (
        train_and_val_list[train_val_split["val"]],
        label_list[train_val_split["val"]],
    )

    learner, val_loss, train_losses, val_losses = fit_and_score_rf(
        train_data, train_labels, val_data, val_labels, hp["n_estimators"]
    )

    print(f"Job {job_id}, Validation Loss: {val_loss:.7f}")

    # Save the model
    model_save_dir = "../models/rf/"
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f"model_{job_id}.pth")
    pickle.dump(learner, open(model_save_path, "wb"))

    return {"objective": -val_loss, "metadata": {"val_loss": val_loss}}


if __name__ == "__main__":
    hp = {"job_id": "testing_rf", "n_estimators": 100}

    run_rf(hp)
