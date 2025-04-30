import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F


class DGLAttentionModule(nn.Module):
    """A graph attention network module implemented using DGL's message passing.

    This module performs multi-head attention on graph-structured data, where attention
    scores are computed between neighboring nodes and used to
    aggregate feature information.
    The implementation follows the standard query-key-value attention mechanism adapted
    for graphs using DGL's efficient message passing primitives.

    Args:
        dim (int): The dimensionality of input and output node features.
        num_heads (int): Number of attention heads. Must divide evenly into `dim`.
        dropout (float): Dropout probability applied to the final output.

    Raises:
        AssertionError: If `dim` is not divisible by `num_heads`.

    Attributes:
        dim (int): Stored dimensionality of node features.
        num_heads (int): Stored number of attention heads.
        head_dim (int): Dimensionality of each attention head (dim // num_heads).
        dropout (float): Stored dropout probability.
        W_query (nn.Linear): Linear projection for queries.
        W_key (nn.Linear): Linear projection for keys.
        W_value (nn.Linear): Linear projection for values.
        W_out (nn.Linear): Final output projection.
        dropout_layer (nn.Dropout): Dropout layer for output regularization.

    Example:
        >>> import dgl
        >>> module = DGLAttentionModule(dim=64, num_heads=4, dropout=0.1)
        >>> graph = dgl.rand_graph(num_nodes=10, num_edges=30)
        >>> x = torch.randn(10, 64)  # 10 nodes with 64-dim features
        >>> output = module(graph, x)  # returns transformed node features
    """

    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.W_query = nn.Linear(dim, dim)
        self.W_key = nn.Linear(dim, dim)
        self.W_value = nn.Linear(dim, dim)
        self.W_out = nn.Linear(dim, dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, graph, x):
        """Forward pass of the graph attention module.

        Args:
            graph (dgl.DGLGraph): The input graph structure.
            x (torch.Tensor): Node feature tensor of shape (num_nodes, dim).

        Returns:
            torch.Tensor: Transformed node features of shape (num_nodes, dim).
        """
        queries = self.W_query(x).view(-1, self.num_heads, self.head_dim)
        keys = self.W_key(x).view(-1, self.num_heads, self.head_dim)
        values = self.W_value(x).view(-1, self.num_heads, self.head_dim)

        graph.ndata["q"] = queries
        graph.ndata["k"] = keys
        graph.ndata["v"] = values

        graph.apply_edges(fn.u_dot_v("q", "k", "score"))
        attention_scores = graph.edata["score"] / (self.head_dim**0.5)
        attention_probs = F.softmax(attention_scores, dim=1)

        graph.edata["attn"] = attention_probs
        graph.update_all(fn.u_mul_e("v", "attn", "m"), fn.sum("m", "h"))
        x = graph.ndata["h"].reshape(-1, self.dim)

        x = self.W_out(x)
        x = self.dropout_layer(x)

        return x
