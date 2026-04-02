# Assignment 2: From Trees to Neural Networks

Comparison of Gradient Boosted Decision Trees (XGBoost) and Multi-Layer Perceptron (MLP) on the UCI Bank Marketing Dataset.

## Dataset
[Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) — 45,211 samples, 16 features, binary classification (term deposit subscription).

## Setup
```bash
pip install ucimlrepo xgboost scikit-learn pandas numpy matplotlib seaborn
```

## Usage
Run all cells top to bottom in `A2_AML.ipynb`.
Dataset is fetched automatically via `ucimlrepo` — no manual download needed.


## Results Summary
| Metric | XGBoost | MLP |
|---|---|---|
| F1 (Yes class) | 0.593 | 0.494 |
| AUC-PR | 0.564 | 0.551 |
| Recall (Yes) | 0.723 | 0.415 |
| Training Time | 7.9s | 30.1s |

## Key Findings
- XGBoost outperforms MLP on F1, AUC-PR, and recall — better suited for this imbalanced tabular dataset
- MLP achieves higher precision (0.61 vs 0.50) — preferable when minimizing false positives matters
- All MLP architectures converge identically, indicating dataset saturation at 38 features
- `poutcome_success` is the strongest predictor by XGBoost feature importance (Gain)
