import os
import csv
import pdb
import pickle
import random
import numpy as np
from sklearn import svm
from sklearn.metrics import mean_squared_error


def dense_and_concat(x, concat=False):
    if not concat:
        return np.asarray(x["feats"].todense())
    else:
        return np.asarray(
            np.concatenate((x["feats"].todense(), x["adj"].todense()), axis=1)
        )


def run_svm(hp):
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
    # pdb.set_trace()
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

    # Fit model
    regressor = svm.SVR(
        kernel="poly", C=hp["C"], gamma=hp["gamma"], degree=hp["degree"]
    )
    regressor.fit(train_data, train_labels)
    output = regressor.predict(val_data)
    val_loss = mean_squared_error(val_labels, output)
    val_losses = [val_loss]
    train_losses = [mean_squared_error(train_labels, regressor.predict(train_data))]

    print(f"Job {job_id}, Validation Loss: {val_loss:.7f}")

    # Save the model
    model_save_dir = "../models/svm/"
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f"model_{job_id}.pth")
    pickle.dump(regressor, open(model_save_path, "wb"))

    # Return the negative final validation loss as the metric for hyperparameter tuning
    return {"objective": -val_loss, "metadata": {"val_loss": val_loss}}

    # return model


if __name__ == "__main__":
    hp = {
        "job_id": "testing_svm",
        "C": 1,
        "gamma": "auto",
        "degree": 3,
    }

    run_svm(hp)
