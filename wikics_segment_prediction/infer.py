import csv
from datetime import datetime
from pathlib import Path

import dvc.api
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from .modules.dgl_attention_module import DGLAttentionModule
from .pl_wrapper import GraphTransformerPL
from .wiki_cs_dataset import WikiCSDataset


class WikiCSInferer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg


def run_testing(cfg: DictConfig):
    device = cfg.settings.device
    torch.set_float32_matmul_precision("high")

    test_idx_path = cfg.settings.test_idx_path
    with dvc.api.open(test_idx_path, remote="data", mode="rb") as f:
        test_idx = torch.LongTensor(pd.read_parquet(f)["index"].values)

    dataset = WikiCSDataset(
        train_idx=None,
        val_idx=None,
        test_idx=test_idx,
        file_path=cfg.train.data_path,
    )

    pl_model = GraphTransformerPL(
        dataset=dataset,
        attention_module=DGLAttentionModule,
        num_layers=cfg.model.num_layers,
        input_dim=dataset.features.size(1),
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        hidden_dim_multiplier=cfg.model.hidden_dim_multiplier,
        dropout=cfg.model.dropout,
        lr=cfg.model.lr,
        device=device,
        num_workers=cfg.train.num_workers,
    ).to(device)

    model_path = Path(cfg.settings.save_dir_model) / cfg.settings.model_name

    with dvc.api.open(str(model_path), remote="models", mode="rb") as saved_model:
        checkpoint = torch.load(saved_model, map_location=device)

    pl_model.model.load_state_dict(checkpoint)
    pl_model.eval()

    trainer = pl.Trainer(
        accelerator="gpu" if device == "cuda" else "cpu",
        enable_progress_bar=True,
        logger=False,
    )

    results = trainer.test(pl_model)

    test_metrics = results[0]
    test_metrics["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    predictions_dir = Path(cfg.settings.get("save_dir_predictions", "predictions"))
    predictions_dir.mkdir(parents=True, exist_ok=True)

    labels_file = predictions_dir / "class_labels.csv"
    with open(labels_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_index", "label"])
        for idx, label in enumerate(dataset.labels):
            writer.writerow([idx, label])
    print(f"Class labels saved to: {labels_file}")
