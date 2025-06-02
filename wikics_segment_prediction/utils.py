import csv
from pathlib import Path


def edges_to_pairwise_matrix(edge_list):
    """
    Convert a list of edges to a pairwise list with
    source and destination nodes.

    Args:
        edge_list: List of lists
            where edge_list[i] contains destination nodes from node i.

    Returns:
        An array with two columns (source, destination) representing all edges.
    """
    pairwise_edges = []
    for source, destinations in enumerate(edge_list):
        for dest in destinations:
            pairwise_edges.append([source, dest])

    return list(pairwise_edges)


def save_metrics_to_csv(metrics: dict, file_path: str):
    """Save metrics dictionary to a CSV file."""
    path = Path(file_path)
    file_exists = path.is_file()

    with open(file_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(metrics)
