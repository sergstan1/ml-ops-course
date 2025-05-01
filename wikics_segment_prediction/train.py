import os

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from .modules.dgl_attention_module import DGLAttentionModule
from .pl_wrapper import GraphTransformerPL
from .wiki_cs_dataset import WikiCSDataset


class WikiCSTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def train(self):
        device = self.cfg.settings.device
        save_dir_plots = self.cfg.settings.save_dir_plots
        save_dir_model = self.cfg.settings.save_dir_model

        train_idx_path = self.cfg.settings.train_idx_path
        val_idx_path = self.cfg.settings.val_idx_path
        test_idx_path = self.cfg.settings.test_idx_path

        experiment_name = OmegaConf.select(
            self.cfg, "mlflow.experiment_name", default="wiki_cs_experiment"
        )
        run_name = OmegaConf.select(self.cfg, "mlflow.run_name", default="run_1")
        tracking_uri = OmegaConf.select(self.cfg, "mlflow.tracking_uri", default=None)

        mlflow_logger = MLFlowLogger(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
        )

        params = OmegaConf.to_container(self.cfg, resolve=True)
        mlflow_logger.log_hyperparams(params)

        dataset = WikiCSDataset(
            train_idx=torch.LongTensor(pd.read_parquet(train_idx_path)["index"].values),
            val_idx=torch.LongTensor(pd.read_parquet(val_idx_path)["index"].values),
            test_idx=torch.LongTensor(pd.read_parquet(test_idx_path)["index"].values),
            file_path=self.cfg.train.data_path,
        )
        model = GraphTransformerPL(
            dataset,
            attention_module=DGLAttentionModule,
            num_layers=self.cfg.model.num_layers,
            input_dim=dataset.features.size(1),
            hidden_dim=self.cfg.model.hidden_dim,
            num_heads=self.cfg.model.num_heads,
            hidden_dim_multiplier=self.cfg.model.hidden_dim_multiplier,
            dropout=self.cfg.model.dropout,
            lr=self.cfg.model.lr,
            device=device,
            num_workers=self.cfg.train.num_workers,
        ).to(device)

        checkpoint_cb = ModelCheckpoint(
            monitor="val_f1",
            mode="max",
            save_top_k=1,
            filename="best_model",
            save_last=True,
        )

        callback = VisualizationCallback(
            save_dir_plots=save_dir_plots, save_dir_model=save_dir_model
        )
        trainer = pl.Trainer(
            max_steps=self.cfg.train.num_steps,
            val_check_interval=1,
            callbacks=[checkpoint_cb, callback],
            accelerator="gpu" if device == "cuda" else "cpu",
            enable_progress_bar=True,
            log_every_n_steps=1,
            logger=mlflow_logger,
        )

        trainer.fit(model)
        trainer.test(model)

        if save_dir_plots is not None:
            model.plot_training_curve(save_dir_plots)
            model.plot_validation_metrics(save_dir_plots)
            model.plot_confusion_matrix("test", save_dir_plots)
            if model.is_binary:
                model.plot_confusion_matrix("val", save_dir_plots)

        return model.model.cpu()


class VisualizationCallback(pl.Callback):
    def __init__(self, save_dir_plots=None, save_dir_model=None):
        self.save_dir_plots = save_dir_plots
        self.save_dir_model = save_dir_model

    def on_train_end(self, trainer, pl_module):
        if self.save_dir_model is not None:
            final_model_path = os.path.join(self.save_dir_model, "final_model.pth")
            torch.save(pl_module.model.state_dict(), final_model_path)

        if self.save_dir_plots is not None:
            pl_module.plot_training_curve(self.save_dir_plots)
            pl_module.plot_validation_metrics(self.save_dir_plots)
            pl_module.plot_confusion_matrix("test", self.save_dir_plots)
            if pl_module.is_binary:
                pl_module.plot_confusion_matrix("val", self.save_dir_plots)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision("high")

    trainer = WikiCSTrainer(cfg)
    trained_model = trainer.train()

    model_path = os.path.join(cfg.settings.save_dir_model, "final_model.pth")
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
