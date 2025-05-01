import os
from datetime import datetime

import dvc.api
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from .model import GraphTransformerPL
from .modules.dgl_attention_module import DGLAttentionModule
from .utils import save_metrics_to_csv
from .wiki_cs_dataset import WikiCSDataset


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    device = cfg.settings.device
    torch.set_float32_matmul_precision("high")

    dataset = WikiCSDataset()

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

    model_path = os.path.join(cfg.settings.save_dir_model, cfg.settings.model_name)

    with dvc.api.open(model_path, remote="models", mode="rb") as saved_model:
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

    metrics_dir = cfg.settings.get("save_dir_metrics", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_file = os.path.join(metrics_dir, "test_metrics.csv")

    save_metrics_to_csv(test_metrics, metrics_file)
    print(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
