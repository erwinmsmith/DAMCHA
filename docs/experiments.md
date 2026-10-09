# Experiments

## Repository workflows

| Component | Entry point | Task |
| --- | --- | --- |
| Decoder and DAMCHA | `models/transformer_damcha.py` | Causal sequence modeling |
| Vision Transformer | `models/vit.py` | Image classification model API |
| CV trainer | `train.py` / `main.py` | Masked patch reconstruction on CIFAR-10/100 and MNIST |
| NLP trainer | `train_nlp.py` / `main.py` | Source-conditioned WMT14 and CommonGen generation |
| Baselines | `models/baselines.py` | THA, DCMHA, ColMHA, MMA and MoA |

## Paper settings

Section 5.1 specifies the following decoder configuration:

| Setting | Manuscript |
| --- | --- |
| Decoder layers | 10 |
| Attention heads | 8 |
| Model / head dimension | 512 / 64 |
| Training epochs | 50 |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 0.0003 |
| Metric MLP | 9 layers |
| Text datasets | WMT14 English–German, CommonGen |
| Image representation | 1024 spatial tokens per 32 × 32 image, 8-bit pixels |

The CLI provides the model-dimension, layer, batch-size and learning-rate controls. `--mlp_hidden_dims` lists the hidden widths; the final output projection adds one layer. Thus eight comma-separated hidden widths define a nine-layer generator. Widths and feed-forward size are experiment configuration choices; Section 5.1 specifies the generator depth.

The repository CV workflow uses masked patch reconstruction. Its default 4 × 4 patches produce 64 tokens per 32 × 32 image. The paper's autoregressive pixel-generation experiment is a separate task specification. The commands here run the existing reconstruction workflow. The current release covers the modules and experiment pipelines present in this repository.

## Run comparisons

```bash
bash scripts/run_all_experiments.sh --dry-run
bash scripts/run_all_experiments.sh
SKIP_DONE=1 bash scripts/run_all_experiments.sh
```

The suite runs DAMCHA, standard MHA and five baselines on four datasets: 28 runs. It uses the CLI's compact configuration (`D=128`, four layers, two hidden MLP layers of widths 256 and 512). Set `PYTHON`, `EPOCHS`, `CV_BATCH`, `NLP_BATCH`, `NUM_WORKERS`, `SEED` or `LOG_DIR` as environment variables. Successful runs receive a `.done` marker; failed runs make the script exit with an error.

## Evaluation

**Images:** pixel MSE, distribution FID from torchvision's ImageNet Inception-v3 features, and squared feature distances for individual image pairs. Feature-distance quartiles describe paired feature errors. Top-MSE subsets are ranked by reconstruction error, which corresponds to fixed-variance Gaussian reconstruction likelihood. Their FID is evaluated as a distribution.

**Text:** target-token NLL under source conditioning, greedy autoregressive generation, and mean log `P(reference | hypothesis)` from the configured BART scorer. BARTScore@Top-50 selects the 50 examples with lowest NLL, then averages their BARTScores. The scorer implementation follows the conditional log-probability convention of [BARTScore](https://github.com/neulab/BARTScore); this repository's scorer checkpoint is `facebook/bart-base`.

CSV files retain per-epoch metrics; JSON files contain per-example metrics. Export each run's best validation epoch:

```bash
python -m scripts.parse_results --input outputs --output logs/experiments/results.csv
```

## Profiling

```bash
python -m scripts.profile_flops_latency --device cuda --batch-size 8 --tokens 64
python -m scripts.profile_memory --device cuda:0 --batch-size 8 --tokens 64
```

Profiling measures the current decoder implementation with the CLI's compact architecture. FLOPs cover matrix-product and convolution operators recognized by `torch.profiler`; latency covers the whole forward pass. The CSV records device, batch size and sequence length. Memory profiling reports CUDA peak allocated MiB for inference and backward passes.

## Regression checks

```bash
pytest -q
```

Tests cover the paper's block-row formula and gradients, input adaptation, prefix causality, shared-generator reuse, parameter registration, baseline masking, conditional generation, EOS handling and complete checkpoint restoration. The selected results on the documentation homepage are manuscript values; local regression checks validate implementation behavior.
