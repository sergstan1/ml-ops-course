import os
import random
from pathlib import Path

import dvc.api
import gensim
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import F1Score

from .wiki_cs_dataset import WikiCSDataset


class DeepWalk:
    def __init__(
        self, graph, walk_length=40, num_walks=10, window=5, embedding_dim=128
    ):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window = window
        self.embedding_dim = embedding_dim

        self.node_ids = []
        self.node_mapping = {}
        for idx, node in enumerate(graph.nodes()):
            if isinstance(node, torch.Tensor):
                scalar = node.item()
            else:
                scalar = node
            self.node_mapping[scalar] = idx
            self.node_ids.append(str(scalar))

    def _random_walk(self, start_node):
        if isinstance(start_node, torch.Tensor):
            current_node = start_node.item()
        else:
            current_node = start_node

        walk = [str(current_node)]
        for _ in range(self.walk_length - 1):
            original_node = self.graph.nodes()[current_node]
            neighbors = self.graph.successors(original_node)

            if len(neighbors) == 0:
                break

            next_node = random.choice(neighbors)
            if isinstance(next_node, torch.Tensor):
                next_scalar = next_node.item()
            else:
                next_scalar = next_node

            walk.append(str(next_scalar))
            current_node = next_scalar

        return walk

    def generate_walks(self):
        walks = []
        nodes = []
        for node in self.graph.nodes():
            if isinstance(node, torch.Tensor):
                nodes.append(node.item())
            else:
                nodes.append(node)

        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self._random_walk(node))
        return walks

    def train_embeddings(self):
        walks = self.generate_walks()
        model = gensim.models.Word2Vec(
            sentences=walks,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=0,
            sg=1,
            workers=os.cpu_count(),
        )

        embeddings = np.zeros((len(self.node_ids), self.embedding_dim))
        for idx, node_str in enumerate(self.node_ids):
            try:
                embeddings[idx] = model.wv[node_str]
            except KeyError:
                embeddings[idx] = np.zeros(self.embedding_dim)
                print(f"Warning: Node {node_str} not found in walks, using zero vector")

        return torch.tensor(embeddings, dtype=torch.float32)


class LinearRegressionPL(pl.LightningModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        input_dim,
        output_dim,
        lr=0.01,
        batch_size=32,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.lr = lr
        self.batch_size = batch_size
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.val_f1 = F1Score(
            task="multiclass", num_classes=output_dim, average="macro"
        )
        self.test_f1 = F1Score(
            task="multiclass", num_classes=output_dim, average="macro"
        )

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_f1.update(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        self.test_f1.update(preds, y)
        return self.test_f1.compute()

    def on_test_epoch_end(self):
        self.log("test_f1", self.test_f1.compute())
        self.test_f1.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class DeepWalkLinearTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.embeddings = None

    def train(self):
        device = self.cfg.settings.device

        train_idx, val_idx, test_idx = self._load_indices()

        dataset = WikiCSDataset(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            file_path=self.cfg.train.data_path,
        )
        labels = dataset.labels.to(device)

        if self.cfg.deepwalk.get("train_embeddings", True):
            self.embeddings = self._train_deepwalk(dataset.graph).to(device)
        else:
            self.embeddings = self._load_embeddings().to(device)

        train_dataset = TensorDataset(self.embeddings[train_idx], labels[train_idx])
        val_dataset = TensorDataset(self.embeddings[val_idx], labels[val_idx])
        test_dataset = TensorDataset(self.embeddings[test_idx], labels[test_idx])

        model = LinearRegressionPL(
            train_dataset,
            val_dataset,
            test_dataset,
            self.embeddings.size(1),
            int(labels.max().item() + 1),
            lr=self.cfg.model.lr,
            batch_size=self.cfg.train.batch_size,
        ).to(device)

        trainer = self._configure_trainer()
        trainer.fit(model)
        trainer.test(model)

        self._save_artifacts(model)
        return model

    def _train_deepwalk(self, graph):
        print("Training DeepWalk embeddings...")
        dw = DeepWalk(
            graph=graph,
            walk_length=self.cfg.deepwalk.walk_length,
            num_walks=self.cfg.deepwalk.num_walks,
            window=self.cfg.deepwalk.window,
            embedding_dim=self.cfg.deepwalk.embedding_dim,
        )
        return dw.train_embeddings()

    def _load_embeddings(self):
        with dvc.api.open(
            self.cfg.deepwalk.embeddings_path, remote="data", mode="rb"
        ) as f:
            df = pd.read_parquet(f)
            return torch.tensor(df.values, dtype=torch.float32)

    def _save_artifacts(self, model):
        if self.cfg.settings.save_dir_model:
            save_dir = Path(self.cfg.settings.save_dir_model)
            save_dir.mkdir(parents=True, exist_ok=True)

            emb_path = save_dir / "deepwalk_embeddings.pt"
            model_path = save_dir / "deepwalk_linear.pth"

            torch.save(self.embeddings, emb_path)
            torch.save(model.state_dict(), model_path)

    def _load_indices(self):
        with dvc.api.open(
            self.cfg.settings.train_idx_path, remote="data", mode="rb"
        ) as f:
            train_idx = pd.read_parquet(f)["index"].values.astype(int)
            train_idx = torch.LongTensor(train_idx)
        with dvc.api.open(
            self.cfg.settings.val_idx_path, remote="data", mode="rb"
        ) as f:
            val_idx = pd.read_parquet(f)["index"].values.astype(int)
            val_idx = torch.LongTensor(val_idx)
        with dvc.api.open(
            self.cfg.settings.test_idx_path, remote="data", mode="rb"
        ) as f:
            test_idx = pd.read_parquet(f)["index"].values.astype(int)
            test_idx = torch.LongTensor(test_idx)
        return train_idx, val_idx, test_idx

    def _configure_trainer(self):
        save_dir = self.cfg.settings.save_dir_model
        if save_dir is not None:
            save_dir = Path(save_dir)

        checkpoint_cb = ModelCheckpoint(
            monitor="val_f1",
            mode="max",
            save_top_k=1,
            dirpath=save_dir,
            filename="deepwalk_best_model",
        )

        logger = self._get_logger()
        return pl.Trainer(
            max_epochs=self.cfg.baseline.num_steps,
            accelerator="gpu" if self.cfg.settings.device == "cuda" else "cpu",
            callbacks=[checkpoint_cb],
            logger=logger,
            enable_progress_bar=True,
            log_every_n_steps=10,
        )

    def _get_logger(self):
        if not self.cfg.mlflow.get("tracking_uri"):
            return None
        return MLFlowLogger(
            experiment_name=self.cfg.mlflow.get(
                "experiment_name", "deepwalk_experiment"
            ),
            run_name=self.cfg.mlflow.get("run_name", "linear_run"),
            tracking_uri=self.cfg.mlflow.tracking_uri,
        )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision("high")
    trainer = DeepWalkLinearTrainer(cfg)
    trainer.train()
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
