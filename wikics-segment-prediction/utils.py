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
