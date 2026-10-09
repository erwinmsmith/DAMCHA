# Getting started

## Install

```bash
git clone https://github.com/erwinmsmith/DAMCHA.git
cd DAMCHA
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Python 3.10+ is supported. Install matching PyTorch and torchvision builds for your CPU or CUDA environment. The test suite runs without dataset or pretrained-model downloads.

## Image reconstruction

CIFAR-10, CIFAR-100 and MNIST are downloaded through torchvision under `data/rawdata/`. The CLI patchifies images and trains a decoder to reconstruct masked patches.

```bash
python main.py --dataset cifar10 --use_M --use_mlp --share_mlp
python main.py --dataset mnist --use_M --use_mlp --share_mlp --device cpu
```

The MNIST CLI selects one input channel automatically. Use `--gpu 1` to choose a CUDA device or `--device cpu` for CPU execution.

## WMT14 English → German

Prepare UTF-8, one-sentence-per-line parallel files from the [WMT14 translation task](https://www.statmt.org/wmt14/translation-task.html). The English and German files must have matching line counts. Use the development split for validation, with preprocessing consistent across splits.

```text
data/rawdata/WMT14/
├── train.en
├── train.de
├── validation.en
└── validation.de
```

```bash
python main.py --dataset wmt --use_M --use_mlp --share_mlp
```

The WMT loader uses `bert-base-multilingual-cased` tokenization. It reads the prepared train and validation splits directly. The decoder conditions on the source sequence and computes loss only on target tokens.

## CommonGen

Prepare [CommonGen](https://github.com/INK-USC/CommonGen) parquet data in the layout consumed by the loader:

```text
data/rawdata/CommonGen/
├── train-00000-of-00001.parquet
└── validation-00000-of-00001.parquet
```

```bash
python main.py --dataset commongen --use_M --use_mlp --share_mlp
```

CommonGen uses `facebook/bart-base` tokenization. The existing CNN/DailyMail loader is also available through `--dataset cnn_dailymail`.

## Save and resume

Training creates `checkpoints/<dataset>/<mode>/`, `outputs/<dataset>/metrics/<mode>/` and visualization directories. Checkpoints contain the attention model, output head, token or patch embeddings, optimizer, scheduler, configuration and random-number states.

```bash
python main.py --dataset cifar10 --use_M --use_mlp --share_mlp \
  --resume checkpoints/cifar10/mlp_shared_depth3_multihead/best_model.pt
```

Use the same model configuration and the actual checkpoint path printed by the run. `--epochs` sets the total training duration, including completed epochs. Checkpoint format 2 accompanies the input-adaptive implementation.

## Documentation

```bash
mkdocs serve
mkdocs build --strict
```

The `Deploy documentation` workflow publishes `docs/` to GitHub Pages on changes to `main`. Site metadata and navigation are configured in `mkdocs.yml`.
