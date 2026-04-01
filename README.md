# Data-Adaptive Mahalanobis Metric Learning for Cross-Head Attention in Transformers

A PyTorch implementation of DAMCHA, a novel attention mechanism that replaces standard Q/K attention with learnable M-matrix based computation.

## Overview

DAMCHA introduces **M-based Attention**, which replaces the standard Q/K attention computation:

- **Standard Attention**: `softmax(Q @ K.T / sqrt(d)) @ V`
- **DAMCHA Attention**: `softmax(X @ M @ X.T / sqrt(d)) @ V`

The M matrix captures cross-head interactions through a structured block design:
- **Diagonal blocks**: Per-head attention patterns
- **Off-diagonal blocks**: Cross-head interactions (learnable linear combinations)

### Multi-head M Mode

In multi-head M mode, off-diagonal blocks are **aggregated to diagonal blocks** before extracting per-head M matrices:

```
For each head h:
  M_h = (M_diag[h] + sum_{j!=h}(M[h,j] + M[j,h])) / (2*n_heads - 1)
```

This allows cross-head information to flow into each head's attention computation.

## Key Features

- **Flexible Attention Modes**: Standard Q/K/V, M0-only, MLP-based M generation
- **Structured M Matrix**: Block-diagonal with learnable off-diagonal interactions
- **Multi-head M Attention**: Aggregate off-diagonal to diagonal, then extract per-head M
- **Off-diagonal Modes**: Direct MLP, linear combination, or Bayesian inference
- **Per-layer Bias**: Low-rank factorized layer-specific M adjustments
- **CV & NLP Support**: Image reconstruction and sequence-to-sequence tasks

## Project Structure

```
DAMCHA/
├── main.py                    # Main entry point
├── train.py                   # CV training (image reconstruction)
├── train_nlp.py               # NLP training (seq2seq)
├── utils.py                   # Utilities (Patchify, MaskGenerator)
├── visualization.py           # Metrics and visualization
├── models/
│   ├── __init__.py
│   ├── m_attention.py         # Core: M-based attention module
│   ├── transformer_damcha.py  # DAMCHA Transformer architecture
│   └── vit.py                 # Vision Transformer with DAMCHA
└── data/
    ├── __init__.py
    ├── datasets.py            # CV datasets (CIFAR, MNIST)
    └── datasets_nlp.py        # NLP datasets (WMT, CommonGen)
```

## Datasets

### CV Datasets
CIFAR-10 and CIFAR-100 are downloaded automatically via `torchvision` on first run and cached under `data/rawdata/`.

### NLP Datasets

| Dataset | Task | Source |
|---------|------|--------|
| **WMT** | Chinese→English Machine Translation | [ModelScope – iic/WMT-Chinese-to-English-Machine-Translation-Training-Corpus](https://www.modelscope.cn/datasets/iic/WMT-Chinese-to-English-Machine-Translation-Training-Corpus) |
| **CommonGen** | Concept-to-Text Generation | [ModelScope – allenai/common_gen](https://www.modelscope.cn/datasets/allenai/common_gen) |

Download the raw files and place them under the corresponding directories:

```
data/rawdata/WMT/        # WMT translation corpus
data/rawdata/CommonGen/  # CommonGen parquet files
```

## Installation

```bash
# Create environment
conda create -n damcha python=3.10
conda activate damcha

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### CV Tasks (Image Generation)

```bash
# Standard Transformer baseline
python main.py --dataset cifar10 --epochs 50

# DAMCHA with shared MLP
python main.py --dataset cifar10 --epochs 50 --use_M --use_mlp --share_mlp

# DAMCHA with multi-head M
python main.py --dataset cifar10 --epochs 50 --use_M --use_mlp --share_mlp --use_multihead_M

# DAMCHA with linear combination off-diagonal
python main.py --dataset cifar10 --epochs 50 --use_M --use_mlp --share_mlp --off_diag_mode linear_comb
```

### NLP Tasks

```bash
# Machine Translation (WMT)
python main.py --dataset wmt --epochs 50 --use_M --use_mlp --share_mlp

# Concept-to-Text (CommonGen)
python main.py --dataset commongen --epochs 50 --use_M --use_mlp --share_mlp
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--d_model` | 128 | Model dimension |
| `--n_heads` | 8 | Number of attention heads |
| `--n_layers` | 4 | Number of decoder layers |
| `--use_M` | False | Enable M-based attention |
| `--use_mlp` | False | Use MLP for M generation |
| `--share_mlp` | False | Share MLP across layers |
| `--off_diag_mode` | 'mlp' | 'mlp', 'linear_comb', 'bayesian' |
| `--off_diag_scale` | 0.5 | Off-diagonal block scale |
| `--use_multihead_M` | False | Per-head M matrices |

## Architecture

### M-based Attention

```
Standard:  Attn = softmax((X @ W_q) @ (X @ W_k).T / sqrt(d)) @ V
DAMCHA:    Attn = softmax(X @ M @ X.T / sqrt(d)) @ V
```

### Structured M Matrix

```
M = | M^(1)   M^(1,2) ... M^(1,H) |
    | M^(2,1) M^(2)   ... M^(2,H) |
    | ...     ...     ... ...     |
    | M^(H,1) M^(H,2) ... M^(H)   |

Off-diagonal: M^(i,j) = sum_k(alpha_{i,j,k} * M^(k))
```

## API

```python
from models import TransformerPC, VisionTransformer

# Create DAMCHA Transformer
model = TransformerPC(
    d_model=128,
    n_heads=8,
    n_layers=4,
    use_M=True,
    use_mlp=True,
    share_mlp=True,
    off_diag_mode='linear_comb',
)

# Get M matrices
m_matrices = model.get_M_matrices()

# Get off-diagonal parameters
off_diag_params = model.get_off_diag_params()
```
