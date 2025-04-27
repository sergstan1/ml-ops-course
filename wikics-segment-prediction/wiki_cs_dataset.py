import json

import dgl
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from .utils import edges_to_pairwise_matrix


class WikiCSDataset:
    """A dataset class for loading and processing the WikiCS graph dataset.

    This class handles loading the dataset
    from a JSON file, creating graph structures,
    and generating train/validation/test splits.
    The dataset contains node features,
    labels, and edges between nodes.

    Args:
        file_path (str): Path to the raw JSON data file.
             Defaults to "data/wikics/raw/data.json".
        test_size (float): Proportion of data to use for
             test & validation splits. Defaults to 0.5.
        val_split (float): Proportion of the test/validation split
             to use for validation. Defaults to 0.5.
        random_state (int): Random
             seed for reproducible splits. Defaults to 42.

    Attributes:
        features (list): Node features from the dataset
        labels (ndarray): Node labels as NumPy array
        edges (ndarray): Edge connections as NumPy array
        train_idx (Tensor): Training node indices
        val_idx (Tensor): Validation node indices
        test_idx (Tensor): Test node indices
        graph (dgl.DGLGraph): Graph representation of the dataset
    """

    def __init__(
        self,
        file_path="data/wikics/raw/data.json",
        test_size=0.5,
        val_split=0.5,
        random_state=42,
    ):
        with open(file_path) as json_data:
            data = json.load(json_data)

        self.features = data["features"]
        self.labels = np.array(data["labels"])
        self.edges = np.array(edges_to_pairwise_matrix(data["links"]))

        full_idx = np.arange(len(self.labels))
        train_idx, val_and_test_idx = train_test_split(
            full_idx,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels,
        )
        val_idx, test_idx = train_test_split(
            val_and_test_idx,
            test_size=val_split,
            random_state=random_state,
            stratify=self.labels[val_and_test_idx],
        )

        self.train_idx = torch.from_numpy(train_idx)
        self.val_idx = torch.from_numpy(val_idx)
        self.test_idx = torch.from_numpy(test_idx)

        self.graph = dgl.graph(
            (self.edges[:, 0], self.edges[:, 1]), num_nodes=len(self.labels)
        )

    @property
    def feature_tensor(self):
        """torch.FloatTensor: Node features converted to float tensor"""
        return torch.FloatTensor(self.features)

    @property
    def label_tensor(self):
        """torch.LongTensor: Node labels converted to long tensor"""
        return torch.LongTensor(self.labels)
