import torch


class ServingModel(torch.nn.Module):
    def __init__(self, model, graph):
        super().__init__()
        self.model = model
        self.graph = graph

    def forward(self, features):
        device = next(self.model.parameters()).device
        self.graph = self.graph.to(device)
        return self.model(self.graph, features)
