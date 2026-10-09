# DAMCHA

**Data-Adaptive Mahalanobis Metric Learning for Cross-Head Attention in Transformers**

**Accepted at NeurIPS 2026**


[Documentation](https://erwinmsmith.github.io/DAMCHA/) · [Method](docs/method.md) · [Experiments](docs/experiments.md) · [Citation](CITATION.bib)

DAMCHA learns an input-dependent cross-head metric for Transformer attention. A metric generator maps the input context to a full matrix, and each head uses the sum of its block row to compute attention. Sharing the generator across the stack gives every layer access to the same family of adaptive metrics.

```text
Input context → mean pooling → metric generator fθ → M(X)
                                                   ↓ RowSum
                          head metrics → attention → concatenate → output
```

This repository provides the DAMCHA attention module, decoder-only Transformers, a Vision Transformer, image reconstruction and conditional text training, attention baselines, and evaluation tools.

## Installation

Use Python 3.10 or newer and a matching PyTorch/torchvision installation for your device.

```bash
git clone https://github.com/erwinmsmith/DAMCHA.git
cd DAMCHA
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Quick start

```python
import torch
from models import TransformerPC

model = TransformerPC(
    d_model=128, n_heads=8, d_ff=256, n_layers=4,
    use_M=True, use_mlp=True, share_mlp=True,
    mlp_hidden=(256, 512),
)
x = torch.randn(2, 16, 128)
y = model(x)  # [batch, tokens, features]
```

Run the existing reconstruction and text workflows:

```bash
# CIFAR-10 masked image reconstruction; dataset downloads automatically.
python main.py --dataset cifar10 --use_M --use_mlp --share_mlp

# WMT14 English → German; prepare the parallel files described in the docs.
python main.py --dataset wmt --use_M --use_mlp --share_mlp

# CommonGen concept-to-text generation.
python main.py --dataset commongen --use_M --use_mlp --share_mlp

# Inspect all 28 experiment commands before running the suite.
bash scripts/run_all_experiments.sh --dry-run
```

See [Getting started](docs/getting-started.md) for dataset layouts, device selection and checkpoint resume. [Experiments](docs/experiments.md) records the paper settings alongside the repository workflows.

## Method

For `d = h × dh`, the generator produces `M(X)` with shape `[batch, d, d]`. Partition it into `h × h` blocks of shape `[dh, dh]`:

```text
M(X)  = fθ(mean(X))
mᵢ(X) = Σⱼ Mᵢⱼ(X)
Aᵢ(X) = softmax(Xᵢ mᵢ(X) Xᵢᵀ / √dh)
Oᵢ(X) = Aᵢ(X) Vᵢ
```

The implementation follows the head-slice formulation in Appendix D. Causal decoding uses a prefix mean for each query; noncausal ViT attention uses the full-sequence mean. The full metric is unconstrained, with optional structural and low-rank ablations. See [Method](docs/method.md) for tensor shapes and code correspondence.

## Repository layout

```text
models/          Attention, decoder, ViT and baseline modules
data/            Dataset loading and tokenization
scripts/         Experiment suite, result export and profiling
tests/           Numerical, causality and training regression tests
docs/            Documentation source
.github/         Test and documentation deployment workflows
main.py          Training CLI
train.py         Image reconstruction training
train_nlp.py     Conditional text training
evaluation_nlp.py NLP evaluation
utils.py         Patch embedding, masking and checkpoints
visualization.py Image metrics and visualizations
CITATION.cff     GitHub citation metadata
CITATION.bib     Paper citation
```

Datasets, checkpoints, logs and generated outputs are created at runtime and excluded from version control.

## Development

```bash
pytest -q
mkdocs build --strict
mkdocs serve
```

## Citation

```bibtex
@inproceedings{duan2026damcha,
  title     = {Data-Adaptive Mahalanobis Metric Learning for Cross-Head Attention in Transformers},
  author    = {Duan, Zhenke and Li, Xin and Pan, Jiqun and {Dong Xiaofei} and Ning, Hanwen and Song, Xinyuan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {Accepted at NeurIPS 2026}
}
```

## License

[MIT](LICENSE).
