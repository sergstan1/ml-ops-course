# WikiCS: Segment Prediction for Computer Science Wikipedia Pages

## Project Overview
This project aims to predict computer science segments in Wikipedia's graph structure (where vertices represent pages and edges represent hyperlinks). The solution can be useful for contextual search within Wikipedia.

**Key Features:**
- Predicts segment membership probabilities for Wikipedia pages
- Works with graph structure embeddings only (no node features)
- Provides baseline and neural network approaches

## Dataset
The WikiCS dataset contains:
- 11,701 vertices (Wikipedia pages)
- 215,863 edges (hyperlinks)
- Average vertex degree: 36.90
- 10 target classes (segments)

Dataset sources:
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/2.5.3/generated/torch_geometric.datasets.WikiCS.html)
- [Papers With Code](https://paperswithcode.com/dataset/wiki-cs)

## Input/Output Format
- **Input:** Wikipedia graph structure
- **Output:** Probability vector (10 dimensions) for each test vertex

## Metrics
Primary evaluation metrics:
- Micro-averaged F1-score
- Macro-averaged F1-score

Target performance: ≥ 0.4 F1-score (baseline performance of logistic regression on graph embeddings)

## Validation
Using standard `train_test_split` with `random_state=42` for reproducibility.

## Modeling Approaches

### Baseline Model
1. Generate Instant Embeddings ([paper reference](https://arxiv.org/abs/2010.06992))
2. Train logistic regression on these embeddings

### Main Model
- PyTorch Lightning neural network
- Architecture:
  - Multiple layers (number to be tuned)
  - ReLU activation functions
- Optimizer: Adam

## Implementation
The final model will be packaged as a library for predicting segments on current Wikipedia data.

## Requirements
- Python 3.x
- PyTorch
- PyTorch Geometric
- PyTorch Lightning
- scikit-learn

## Usage
```python
# Example usage will be provided after implementation
from wikics_predictor import SegmentPredictor

model = SegmentPredictor.load_pretrained()
predictions = model.predict(wiki_graph)
