from pathlib import Path

import dgl
import dvc.api
import mlflow.pytorch
import pandas as pd
import pytorch_lightning as pl
import torch
from mlflow.models.signature import ModelSignature
from mlflow.pytorch import get_default_pip_requirements
from mlflow.types.schema import ColSpec, Schema
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from .modules.dgl_attention_module import DGLAttentionModule
from .modules.serving_model import ServingModel
from .pl_wrapper import GraphTransformerPL
from .wiki_cs_dataset import WikiCSDataset


class WikiCSTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def train(self):
        device = self.cfg.settings.device
        save_dir_plots = (
            Path(self.cfg.settings.save_dir_plots)
            if self.cfg.settings.save_dir_plots
            else None
        )
        save_dir_model = (
            Path(self.cfg.settings.save_dir_model)
            if self.cfg.settings.save_dir_model
            else None
        )

        if save_dir_plots:
            save_dir_plots.mkdir(parents=True, exist_ok=True)
        if save_dir_model:
            save_dir_model.mkdir(parents=True, exist_ok=True)

        train_idx, val_idx, test_idx = self._load_indices()

        experiment_name = OmegaConf.select(
            self.cfg, "mlflow.experiment_name", default="wiki_cs_experiment"
        )
        run_name = OmegaConf.select(
            self.cfg, "mlflow.run_name", default="run_without_a_name"
        )
        tracking_uri = OmegaConf.select(self.cfg, "mlflow.tracking_uri", default=None)

        params = OmegaConf.to_container(self.cfg, resolve=True)
        mlflow_logger = None

        if tracking_uri is not None:
            mlflow_logger = MLFlowLogger(
                experiment_name=experiment_name,
                run_name=run_name,
                tracking_uri=tracking_uri,
            )
            mlflow_logger.log_hyperparams(params)

        dataset = WikiCSDataset(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
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

        trainer, checkpoint_cb = self._configure_trainer(
            tracking_uri,
            save_dir_plots,
            save_dir_model,
            device,
            mlflow_logger,
            dataset,
        )

        trainer.fit(model)
        trainer.test(model)

        if mlflow_logger is not None:
            if checkpoint_cb.best_model_path:
                try:
                    best_model_wrapper = GraphTransformerPL.load_from_checkpoint(
                        checkpoint_path=checkpoint_cb.best_model_path,
                        dataset=dataset,
                        attention_module=DGLAttentionModule,
                        num_layers=self.cfg.model.num_layers,
                        input_dim=dataset.features.size(1),
                        hidden_dim=self.cfg.model.hidden_dim,
                        num_heads=self.cfg.model.num_heads,
                        hidden_dim_multiplier=self.cfg.model.hidden_dim_multiplier,
                        dropout=self.cfg.model.dropout,
                        lr=self.cfg.model.lr,
                        device="cpu",
                        num_workers=self.cfg.train.num_workers,
                    )
                    best_pure_model = best_model_wrapper.model

                    input_schema = Schema(
                        [
                            ColSpec(type="double", name="node_features"),
                        ]
                    )
                    output_schema = Schema(
                        [
                            ColSpec(type="double", name="logits"),
                        ]
                    )
                    signature = ModelSignature(
                        inputs=input_schema, outputs=output_schema
                    )

                    serving_model = ServingModel(best_pure_model, dataset.graph)

                    default_reqs = get_default_pip_requirements()
                    dgl_version = dgl.__version__
                    pip_reqs = [
                        f"dgl=={dgl_version}" if "dgl" in req else req
                        for req in default_reqs
                    ]

                    with mlflow.start_run(run_id=mlflow_logger.run_id):
                        mlflow.pytorch.log_model(
                            pytorch_model=serving_model,
                            artifact_path="best_model",
                            signature=signature,
                            pip_requirements=pip_reqs,
                        )
                except Exception as exception_name:
                    print(f"Error logging best model to MLflow: {str(exception_name)}")

            if save_dir_model and save_dir_model.exists():
                try:
                    with mlflow.start_run(run_id=mlflow_logger.run_id):
                        mlflow.log_artifacts(
                            local_dir=str(save_dir_model),
                            artifact_path="final_model_artifacts",
                        )
                except Exception as exception_name:
                    print(f"Error logging artifacts to MLflow: {str(exception_name)}")

        return model.model.cpu()

    def _load_indices(self):
        with dvc.api.open(
            self.cfg.settings.train_idx_path, remote="data", mode="rb"
        ) as f:
            train_idx = torch.LongTensor(pd.read_parquet(f)["index"].values)
        with dvc.api.open(
            self.cfg.settings.val_idx_path, remote="data", mode="rb"
        ) as f:
            val_idx = torch.LongTensor(pd.read_parquet(f)["index"].values)
        with dvc.api.open(
            self.cfg.settings.test_idx_path, remote="data", mode="rb"
        ) as f:
            test_idx = torch.LongTensor(pd.read_parquet(f)["index"].values)
        return train_idx, val_idx, test_idx

    def _configure_trainer(
        self,
        tracking_uri,
        save_dir_plots,
        save_dir_model,
        device,
        mlflow_logger,
        dataset,
    ):
        checkpoint_cb = ModelCheckpoint(
            monitor="val_f1",
            mode="max",
            save_top_k=1,
            filename="best_model",
            save_last=True,
        )

        callback = VisualizationCallback(
            save_dir_plots=save_dir_plots,
            save_dir_model=save_dir_model,
            dataset=dataset,
        )

        if tracking_uri is not None:
            trainer = pl.Trainer(
                max_steps=self.cfg.train.num_steps,
                val_check_interval=1,
                callbacks=[checkpoint_cb, callback],
                accelerator="gpu" if device == "cuda" else "cpu",
                enable_progress_bar=True,
                log_every_n_steps=1,
                logger=mlflow_logger,
            )
        else:
            trainer = pl.Trainer(
                max_steps=self.cfg.train.num_steps,
                val_check_interval=1,
                callbacks=[checkpoint_cb, callback],
                accelerator="gpu" if device == "cuda" else "cpu",
                enable_progress_bar=True,
                log_every_n_steps=1,
            )

        return trainer, checkpoint_cb


class VisualizationCallback(pl.Callback):
    def __init__(self, save_dir_plots=None, save_dir_model=None, dataset=None):
        self.save_dir_plots = Path(save_dir_plots) if save_dir_plots else None
        self.save_dir_model = Path(save_dir_model) if save_dir_model else None
        self.dataset = dataset
        self.original_device = None

    def on_train_end(self, trainer, pl_module):
        self.original_device = next(pl_module.model.parameters()).device

        if self.save_dir_plots:
            self.save_dir_plots.mkdir(parents=True, exist_ok=True)
            pl_module.plot_training_curve(str(self.save_dir_plots))
            pl_module.plot_validation_metrics(str(self.save_dir_plots))
            pl_module.plot_confusion_matrix("test", str(self.save_dir_plots))
            if pl_module.is_binary:
                pl_module.plot_confusion_matrix("val", str(self.save_dir_plots))

        if self.save_dir_model:
            self.save_dir_model.mkdir(parents=True, exist_ok=True)
            final_model_path = self.save_dir_model / "final_model.pth"
            torch.save(pl_module.model.state_dict(), final_model_path)

            onnx_path = self.save_dir_model / "model.onnx"
            try:
                import warnings

                from torch.jit import TracerWarning

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=TracerWarning)

                    graph_cpu = self.dataset.graph.to("cpu")
                    num_nodes = graph_cpu.number_of_nodes()
                    feature_dim = self.dataset.features.size(1)
                    dummy_input = torch.randn(num_nodes, feature_dim).to("cpu")
                    pl_module.model.graph = pl_module.model.graph.to("cpu")

                    torch.onnx.export(
                        pl_module.model.to("cpu"),
                        dummy_input,
                        onnx_path,
                        input_names=["node_features"],
                        output_names=["logits"],
                    )
                    print(f"ONNX model saved to {onnx_path}")

                    edges_src, edges_dst = self.dataset.graph.edges()
                    torch.save(edges_src, self.save_dir_model / "edges_src.pt")
                    torch.save(edges_dst, self.save_dir_model / "edges_dst.pt")
            except Exception as e:
                print(f"ONNX export failed: {e}")
            finally:
                pl_module.model.to(self.original_device)
                pl_module.model.graph = pl_module.model.graph.to(self.original_device)
                self.dataset.graph = self.dataset.graph.to(self.original_device)


def run_training(cfg: DictConfig):
    torch.set_float32_matmul_precision("high")

    trainer = WikiCSTrainer(cfg)
    trained_model = trainer.train()

    model_path = Path(cfg.settings.save_dir_model) / "final_model.pth"
    if model_path.parent:
        model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
