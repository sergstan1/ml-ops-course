# WikiCS: Segment Prediction for Computer Science Wikipedia Pages

## Project Overview

This project focuses on predicting the segment membership of computer science-related Wikipedia pages using a graph-based structure. Each node represents a Wikipedia page, and each edge represents a hyperlink between pages. The main goal is to classify each page into one of the defined CS topic segments.

**Key Features:**

- Predicts labels for each segment/class using custom GNN
- Visualizations of training results and segment distributions
- Modular and extendable architecture

## Dataset

The project uses the [WikiCS dataset](https://github.com/pmernyei/wiki-cs-dataset), which consists of:

- 11,701 Wikipedia pages as nodes
- 215,000 edges representing hyperlinks
- 10 semantic segments (classes) related to computer science disciplines

The dataset can be accessed using `torch_geometric.datasets.WikiCS`. Note that it includes 20 distinct training/validation/test splits for robust evaluation.

## Setup

### Requirements

- Python 3.12
- [Poetry](https://python-poetry.org/docs/) for environment management
- PyTorch
- PyTorch Geometric
- DGL (Deep Graph Library)
- DVC (optional, for data version control)

### Installation Steps

1. Clone the repository:

```bash
git clone https://github.com/sergstan1/wikics-segment-prediction.git
cd wikics-segment-prediction
```

2. Install dependencies using Poetry:

```bash
poetry install
```

3. Activate the virtual environment:

```bash
poetry shell
```

4. Pull data and models via DVC:

```bash
dvc pull
```

## Train

To train a model from scratch, follow these steps:

### 1. Data Loading

The dataset is downloaded using a path to a json file with features and edge indices.
Also you need to set a hydra config with path to train, val, test indices, stored as a pd.DataFrame with "index" column.

### 2. Training the Model

To train the default model:

```bash
poetry run python3 -m wikics_segment_prediction.train
```

If you want to run with a specific configuration, e.g. path to your data:

```bash
poetry run python3 -m wikics_segment_prediction.train train.data_path=<data_path>
```

Training also supports mlflow support, however, if you do not want to use it set mlflow.tracking_uri to null.
To set mlflow uri you can run:

```bash
poetry run python3 -m wikics_segment_prediction.train mlflow.tracking_uri=<your_uri>
```

## Infer

After training the model, you can use it for inference on new data.
Set a path to test data in hydra configuration, the same as in train.

### Running Inference

```bash
poetry run python3 -m wikics_segment_prediction.infer
```

This will output predictions for each class per node.

## Contact

Project author: [sergstan1](https://github.com/sergstan1)
