"""Graph pipeline helpers exposed through the graph capability package."""


def build_spatial_graph(
        data,
        distType="BallTree",
        k=15,
        rad_cutoff=150,
    ):
    """
    Build spatial graph from coordinates.

    This is the recommended entry point for graph construction.

    Parameters
    ----------
    data : numpy.ndarray
        Spatial coordinates with shape (n_spots, 2).
    distType : str, optional
        Distance type for neighbor search. Default: "BallTree".
    k : int, optional
        Number of neighbors. Default: 15.
    rad_cutoff : float, optional
        Radius cutoff for "Radius" distance type. Default: 150.

    Returns
    -------
    dict
        Graph dictionary with keys:
        - 'adj_norm': normalized adjacency matrix
        - 'adj_label': adjacency matrix with self-loops
        - 'norm_value': normalization value
        - 'edge_index': edge index in COO format
    """
    from ..gr.graph import graph
    g = graph(data, distType=distType, k=k, rad_cutoff=rad_cutoff)
    return g.main()


__all__ = ["build_spatial_graph"]
