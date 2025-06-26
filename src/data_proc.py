import os
import csv
import pdb
import pickle
import torch
import random
import numpy as np
from scipy import sparse
import torch_geometric as pyg
from torch_geometric.utils.convert import to_scipy_sparse_matrix


def zero_pad_batch(batch, target_shape, constant_value):
    padded_batch = []
    for array in batch:
        padded_array = np.pad(
            array,
            (0, target_shape - array.shape[0]),
            mode="constant",
            constant_values=constant_value,
        )
        padded_batch.append(padded_array)

    return np.array(padded_batch)


def process_data():
    # Load train_and_val_graph_list from pickle file
    # with open('../Data/train_and_val_graph_list.pkl', 'rb') as f:
    with open("../Data/test_graph_list.pkl", "rb") as f:
        train_and_val_graph_list = pickle.load(f)

    features = []
    adjacency = []
    label = []
    N = len(train_and_val_graph_list)
    max_feat_dim = 0
    max_adj_dim = 0

    for graph in train_and_val_graph_list:
        # Node features
        features.append(np.reshape(np.asarray(graph.x), (-1,)))
        feat_dim = features[-1].shape[0]
        if feat_dim > max_feat_dim:
            max_feat_dim = feat_dim

        # Flattened upper triangle of adjacency
        temp = np.asarray(to_scipy_sparse_matrix(graph.edge_index).todense())
        adjacency.append(temp[np.triu_indices(temp.shape[0])])
        adj_dim = adjacency[-1].shape[0]
        if adj_dim > max_adj_dim:
            max_adj_dim = adj_dim

        # Labels
        label.append(np.asarray(graph.y))

    max_feat_dim = 808
    max_adj_dim = 20503
    features = zero_pad_batch(features, max_feat_dim, 0)
    adjacency = zero_pad_batch(adjacency, max_adj_dim, 0)
    label = np.squeeze(label)

    # pickle.dump({'feats': sparse.csr_matrix(features), 'adj': sparse.csr_matrix(adjacency),'labels':label},open('../Data/train_and_val_list.pkl','wb'))
    pickle.dump(
        {
            "feats": sparse.csr_matrix(features),
            "adj": sparse.csr_matrix(adjacency),
            "labels": label,
        },
        open("../Data/test_list.pkl", "wb"),
    )


if __name__ == "__main__":
    process_data()
