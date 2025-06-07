# WikiCS: Segment Prediction for Computer Science Wikipedia Pages

## Project Overview

This project focuses on predicting the segment membership of computer science-related Wikipedia pages using a graph-based structure. Each node represents a Wikipedia page, and each edge represents a hyperlink between pages. The main goal is to classify each page into one of the defined CS topic segments.

### Key Features

* Predicts labels for each segment/class using a custom GNN based on DGL
* Visualization of model performance and segment distributions
* Support for training, inference, and production-ready export (e.g., ONNX)
* Modular and extendable codebase

## Dataset

The project uses the [WikiCS dataset](https://github.com/pmernyei/wiki-cs-dataset), which consists of:

* 11,701 Wikipedia pages as nodes
* 215,000 edges representing hyperlinks
* 10 semantic segments (classes) related to computer science disciplines

The dataset is accessible via `torch_geometric.datasets.WikiCS`, with 20 predefined train/val/test splits.

---

## Technical Details

### Setup

#### Requirements

* Python 3.12
* [Poetry](https://python-poetry.org/docs/) (for dependency management)
* PyTorch
* DGL (Deep Graph Library)
* PyTorch Lightning
* Hydra
* MLflow (optional)
* DVC (for data versioning)

#### Installation

1. Clone the repository:

```bash
git clone https://github.com/sergstan1/wikics-segment-prediction.git
cd wikics-segment-prediction
```

2. Install dependencies:

```bash
poetry install
```

3. Activate the environment:

```bash
poetry shell
```

4. (Optional) Pull data and artifacts using DVC:

```bash
dvc pull
```

---

### Train

To train the model, follow these steps:

#### 1. Prepare Data and Config

Ensure your data is formatted as:

* `data_path`: JSON with features and edge indices
* Index files (`train_idx_path`, `val_idx_path`, `test_idx_path`): parquet files with a column named `index`

Configure paths in `conf/train.yaml` or override via CLI.

#### 2. Run Training

Default run:

```bash
poetry run python3 -m wikics_segment_prediction.train
```

With specific config overrides:

```bash
poetry run python3 -m wikics_segment_prediction.train \
  train.data_path=<path_to_data> \
  settings.train_idx_path=<path_to_train_idx> \
  settings.val_idx_path=<path_to_val_idx> \
  settings.test_idx_path=<path_to_test_idx>
```

To enable MLflow tracking:

```bash
poetry run python3 -m wikics_segment_prediction.train \
  mlflow.tracking_uri=http://<mlflow-uri> \
  mlflow.experiment_name=wiki_cs_experiment
```

---

### Production Preparation

After training:

* The best model is saved as `final_model.pth`
* Additional plots and metrics are saved in `save_dir_plots`
* The model is exported to ONNX format as `model.onnx`
* Graph structure saved as `edges_src.pt` and `edges_dst.pt`

ONNX export includes:

* `node_features` input
* `logits` output

Logged to MLflow with dependencies if configured.

---

### Infer

Inference assumes a trained model and valid test data.

```bash
poetry run python3 -m wikics_segment_prediction.infer
```

Required configuration:

* Same as training: `data_path`, `test_idx_path`
* Output: predictions per node saved to file or stdout depending on config

#### Inference Input Format

* Features: node feature matrix (same as training)
* Graph: adjacency as edge index
* Indices: test node indices in parquet with "index" column

---

## Contact

Project author: [sergstan1](https://github.com/sergstan1)
