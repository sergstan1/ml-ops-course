import torch.nn as nn

from .feed_forward_module import FeedForwardModule


class TransformerModule(nn.Module):
    """A transformer module for graph-structured data,
      combining attention and feed-forward layers.

    Args:
        attention_module (nn.Module): Attention module class (e.g., DGLAttentionModule)
        dim (int): Dimension of input features
        num_heads (int): Number of attention heads
        dim_multiplier (int): Multiplier for feed-forward hidden dimension
        dropout (float): Dropout probability

    Shapes:
        Input: (N, dim) where N is number of nodes
        Output: (N, dim)
    """

    def __init__(self, attention_module, dim, num_heads, dim_multiplier, dropout):
        super().__init__()
        self.attention_module = attention_module(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.feed_forward = FeedForwardModule(dim, dim_multiplier, dropout)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, graph, x):
        """Applies transformer operations on graph nodes.

        Args:
            graph (DGLGraph): Input graph
            x (Tensor): Node features

        Returns:
            Tensor: Transformed node features
        """
        residual = x
        x = self.norm1(x)
        x = self.attention_module(graph, x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + x

        return x
