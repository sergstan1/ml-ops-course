import json
import typing as tp

import dgl
import dvc.api
import numpy as np
import torch

from .utils import edges_to_pairwise_matrix


class WikiCSDataset:
    """A dataset class for loading and processing the WikiCS graph dataset.

    This class handles loading the WikiCS dataset from DVC storage and converting it into
    a DGL graph with node features, labels, and optional predefined splits. The dataset
    represents Wikipedia Computer Science articles
    as nodes with links between them as edges.

    Args:
        train_idx (Optional[torch.LongTensor]): Indices of nodes for training.
            If None, no training split will be available.
        val_idx (Optional[torch.LongTensor]): Indices of nodes for validation.
            If None, no validation split will be available.
        test_idx (Optional[torch.LongTensor]): Indices of nodes for testing.
            If None, no test split will be available.
        file_path (str): Path to the raw JSON data file in DVC storage.
            Defaults to "data/wikics/raw/data.json".

    Attributes:
        features (torch.FloatTensor): Node features tensor
          of shape (num_nodes, feature_dim)
        labels (torch.LongTensor): Node labels tensor of shape (num_nodes,)
        edges (np.ndarray): Edge connections array of shape (num_edges, 2)
        train_idx (Optional[torch.LongTensor]): Training node indices (if provided)
        val_idx (Optional[torch.LongTensor]): Validation node indices (if provided)
        test_idx (Optional[torch.LongTensor]): Test node indices (if provided)
        graph (dgl.DGLGraph): Graph representation containing nodes and edges

    Example:
        >>> # With predefined splits
        >>> dataset = WikiCSDataset(train_idx, val_idx, test_idx)
        >>> # Without any splits
        >>> dataset = WikiCSDataset(None, None, None)
        >>> # Accessing graph data
        >>> graph = dataset.graph
        >>> features = dataset.features
    """

    def __init__(
        self,
        train_idx: tp.Optional[torch.LongTensor],
        val_idx: tp.Optional[torch.LongTensor],
        test_idx: tp.Optional[torch.LongTensor],
        file_path: str = "data/wikics/raw/data.json",
    ):
        with dvc.api.open(file_path, remote="data") as json_data:
            data = json.load(json_data)

        self.features = torch.FloatTensor(data["features"])
        self.labels = torch.LongTensor(np.array(data["labels"]))
        self.edges = np.array(edges_to_pairwise_matrix(data["links"]))

        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

        self.graph = dgl.graph(
            (self.edges[:, 0], self.edges[:, 1]), num_nodes=len(self.labels)
        )
