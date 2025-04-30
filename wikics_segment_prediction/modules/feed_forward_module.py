import torch.nn as nn


class FeedForwardModule(nn.Module):
    """A simple feed-forward neural network module with dropout.

    This module consists of two linear transformations with a ReLU activation
    and dropout in between. The hidden dimension is typically a multiple of
    the input dimension.

    Args:
        dim: Input and output dimension of the module.
        dim_multiplier: Multiplier for the hidden dimension (default: 4).
        dropout: Dropout probability (default: 0.2).
    """

    def __init__(self, dim, dim_multiplier=4, dropout=0.2):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * dim_multiplier)
        self.dropout1 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(dim * dim_multiplier, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass of the feed-forward module.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Output tensor of shape (..., dim).
        """
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x
