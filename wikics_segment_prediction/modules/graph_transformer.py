import torch.nn as nn

from .transformer_module import TransformerModule


class GraphTransformer(nn.Module):
    """A graph transformer model composed of multiple transformer layers.

    Args:
        attention_module (nn.Module): The attention module class to use
        num_layers (int): Number of transformer layers
        input_dim (int): Dimension of input features
        hidden_dim (int): Dimension of hidden representations
        num_heads (int): Number of attention heads
        hidden_dim_multiplier (int): Multiplier for FFN hidden dimension
        num_classes (int): Number of output classes (2 for binary classification)
        dropout (float): Dropout probability

    Shapes:
        Input: (N, input_dim) where N is number of nodes
        Output: (N,) for binary classification, (N, num_classes) otherwise
    """

    def __init__(
        self,
        attention_module,
        num_layers,
        input_dim,
        hidden_dim,
        num_heads,
        hidden_dim_multiplier,
        num_classes,
        dropout,
        graph,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerModule(
                    attention_module,
                    hidden_dim,
                    num_heads,
                    hidden_dim_multiplier,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

        if num_classes == 2:
            self.output_head = nn.Linear(hidden_dim, 1)
        else:
            self.output_head = nn.Linear(hidden_dim, num_classes)
        self.graph = graph

    def forward(self, x):
        """Forward pass of the graph transformer.

        Args:
            x (Tensor): Node features of shape (N, input_dim)

        Returns:
            Tensor: Output logits (N,) for binary or (N, num_classes) for multiclass
        """
        x = self.input_proj(x)
        for block in self.transformer_blocks:
            x = block(self.graph, x)
        x = self.final_norm(x)
        logits = self.output_head(x)

        return logits
