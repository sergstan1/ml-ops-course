from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.optim import Adam

from .modules.graph_transformer import GraphTransformer
from .wiki_cs_dataset import WikiCSDataset


class GraphTransformerPL(pl.LightningModule):
    def __init__(
        self,
        dataset: WikiCSDataset,
        attention_module,
        num_layers,
        input_dim,
        hidden_dim,
        num_heads,
        hidden_dim_multiplier,
        dropout,
        lr,
        device,
        num_workers=1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["dataset"])

        self._device = device

        self.graph = dataset.graph.to(device)
        self.features = dataset.features.to(device)
        self.labels = dataset.labels.to(device)
        if dataset.train_idx is not None:
            self.train_idx = dataset.train_idx.to(device)
        if dataset.val_idx is not None:
            self.val_idx = dataset.val_idx.to(device)
        if dataset.test_idx is not None:
            self.test_idx = dataset.test_idx.to(device)

        self.num_classes = int(self.labels.max().item() + 1)
        self.is_binary = self.num_classes == 2

        self.model = GraphTransformer(
            attention_module,
            num_layers,
            input_dim,
            hidden_dim,
            num_heads,
            hidden_dim_multiplier,
            self.num_classes,
            dropout,
        )

        self.criterion = (
            nn.BCEWithLogitsLoss() if self.is_binary else nn.CrossEntropyLoss()
        )

        self.train_losses = []
        self.val_metrics = {"f1": [], "accuracy": [], "auc": []}
        self.test_metrics = {}

        self.best_val_f1 = -np.inf
        self.best_model_state = None

        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.num_workers = num_workers

    def forward(self, graph, features):
        return self.model(graph, features)

    def training_step(self, batch, batch_idx):
        logits = self(self.graph, self.features)
        loss = self._calculate_loss(logits, self.train_idx)
        self.log("train_loss", loss, prog_bar=True)
        self.train_losses.append(loss.detach().cpu().item())
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(self.graph, self.features)
        self.validation_step_outputs.append(
            {"logits": logits[self.val_idx], "labels": self.labels[self.val_idx]}
        )
        return {"logits": logits[self.val_idx], "labels": self.labels[self.val_idx]}

    def on_validation_epoch_end(self):
        logits = self.validation_step_outputs[-1]["logits"]
        labels = self.validation_step_outputs[-1]["labels"]
        metrics = self._calculate_metrics(logits, labels)

        self.log("val_f1", metrics["f1"], prog_bar=True)
        self.log("val_acc", metrics["accuracy"])
        if self.is_binary:
            self.log("val_auc", metrics["auc"])

        if metrics["f1"] > self.best_val_f1:
            self.best_val_f1 = metrics["f1"]
            self.best_model_state = deepcopy(self.state_dict())

        self.val_metrics["f1"].append(metrics["f1"])
        self.val_metrics["accuracy"].append(metrics["accuracy"])
        if self.is_binary:
            self.val_metrics["auc"].append(metrics["auc"])

    def test_step(self, batch, batch_idx):
        logits = self(self.graph, self.features)
        self.test_step_outputs.append(
            {"logits": logits[self.test_idx], "labels": self.labels[self.test_idx]}
        )
        return {"logits": logits[self.test_idx], "labels": self.labels[self.test_idx]}

    def on_test_epoch_end(self):
        logits = self.test_step_outputs[-1]["logits"]
        labels = self.test_step_outputs[-1]["labels"]
        metrics = self._calculate_metrics(logits, labels)

        self.test_metrics = metrics
        self.log("test_f1", metrics["f1"])
        self.log("test_acc", metrics["accuracy"])
        if self.is_binary:
            self.log("test_auc", metrics["auc"])

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)

    def _calculate_loss(self, logits, indices):
        if self.is_binary:
            return self.criterion(
                logits[indices].squeeze(), self.labels[indices].float()
            )
        return self.criterion(logits[indices], self.labels[indices].long())

    def _calculate_metrics(self, logits, labels):
        labels_np = labels.cpu().numpy()
        metrics = {}

        if self.is_binary:
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            metrics["auc"] = roc_auc_score(labels_np, probs)
        else:
            preds = logits.argmax(dim=1).cpu().numpy()

        metrics["f1"] = f1_score(
            labels_np, preds, average="binary" if self.is_binary else "macro"
        )
        metrics["accuracy"] = accuracy_score(labels_np, preds)
        return metrics

    def plot_training_curve(self, save_dir=None):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Training Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        self._save_plot(plt, "training_curve.png", save_dir)

    def plot_validation_metrics(self, save_dir=None):
        plt.figure(figsize=(12, 6))
        plt.plot(self.val_metrics["f1"], label="F1 Score")
        plt.plot(self.val_metrics["accuracy"], label="Accuracy")
        if self.is_binary:
            plt.plot(self.val_metrics["auc"], label="AUC")
        plt.xlabel("Validation Epoch")
        plt.ylabel("Score")
        plt.title("Validation Metrics")
        plt.legend()
        self._save_plot(plt, "validation_metrics.png", save_dir)

    def plot_confusion_matrix(self, split="test", save_dir=None):
        idx = getattr(self, f"{split}_idx").to(self._device)
        logits = self(self.graph.to(self._device), self.features.to(self._device))[idx]
        labels = self.labels[idx].cpu().numpy()

        if self.is_binary:
            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
        else:
            preds = logits.argmax(dim=1).cpu().numpy()

        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{split.capitalize()} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        self._save_plot(plt, f"{split}_confusion_matrix.png", save_dir)

    def _save_plot(self, plt, filename, save_dir):
        if save_dir:
            Path(save_dir).mkdir(exist_ok=True, parents=True)
            plt.savefig(Path(save_dir) / filename)
            plt.close()
        else:
            plt.show()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            [0], batch_size=1, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            [0], batch_size=1, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            [0], batch_size=1, num_workers=self.num_workers
        )
