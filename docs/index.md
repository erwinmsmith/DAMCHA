<div class="hero">
<p class="venue">Accepted at NeurIPS 2026</p>
<h1>Attention that adapts its geometry.</h1>
<p>DAMCHA learns an input-dependent Mahalanobis-like metric across attention heads, combining cross-head interactions with stack-wise parameter sharing.</p>
</div>

**Data-Adaptive Mahalanobis Metric Learning for Cross-Head Attention in Transformers**


[Get started](getting-started.md) · [Explore the method](method.md) · [View the code](https://github.com/erwinmsmith/DAMCHA) · [Cite the paper](citation.md)

## Three core ideas

| Component | Role |
| --- | --- |
| Input-adaptive metric | A mean-pooled context conditions the metric generator. |
| Cross-head interaction | Each effective head metric sums one row of cross-head blocks. |
| Stack-wise sharing | One generator supplies metrics to the Transformer stack. |

## Code and experiments

The repository contains decoder-only DAMCHA models, a Vision Transformer, five attention baselines, masked image reconstruction, conditional text generation, evaluation and profiling tools. The documentation maps these components to the paper and describes their executable configurations.

## Selected paper results

Values below are reported in Table 1 of the manuscript.

| Dataset | Metric | MHA | DAMCHA |
| --- | --- | ---: | ---: |
| CIFAR-10 | FID@Top-50 ↓ | 91.4320 | **63.0295** |
| CIFAR-100 | FID@Top-50 ↓ | 59.6860 | **53.7560** |
| WMT | BARTScore ↑ | −8.4740 | **−5.4451** |
| CommonGen | BARTScore ↑ | −5.9250 | **−5.6142** |

See [Experiments](experiments.md) for task definitions, model settings and evaluation conventions.
