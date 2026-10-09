# Method

DAMCHA stands for **Data-Adaptive Mahalanobis Cross-Head Attention**. It defines attention through a context-dependent bilinear metric.

## Input-conditioned metric

For input `X` of shape `[B, T, D]`, compute a mean context of shape `[B, D]`. An MLP maps this context to a flattened `D × D` cross-head matrix. With `H` heads and `dh = D/H`, reshape the matrix into `[B, H, dh, H, dh]`.

The standard path learns every block directly. Both diagonal and off-diagonal scales equal one. The bilinear matrix has no symmetry or positive-definiteness constraint. Low-rank factorization and per-layer bias are available as ablations.

## RowSum and head scores

Equation (4) defines `mᵢ = Σⱼ Mᵢⱼ`. Sum the block-column dimension to produce `[B, H, dh, dh]`. Appendix D uses the positional head slice `Xᵢ = X[..., i*dh:(i+1)*dh]` in the score equation:

```text
Sᵢ = Xᵢ mᵢ Xᵢᵀ / √dh
Aᵢ = softmax(Sᵢ + mask)
Oᵢ = Aᵢ Vᵢ
Y  = concat(O₁, …, Oₕ) Wₒ
```

Value and output projections retain the standard multi-head interface. The implementation uses the per-head tensor dimensions specified in Appendix D for the quadratic form in Section 3.3.

## Causal context

In an autoregressive decoder, query `t` conditions its metric on `mean(X[:, :t+1])`. This keeps the generator and attention scores causal. ViT uses a full-sequence mean. Padding tokens contribute to neither the context mean nor the attention key set.

The prefix convention is the repository's autoregressive implementation of input conditioning. The single-context complexity expression in Appendix D applies to the full-sequence mean; causal prefix generation has a context for each query.

## Stack-wise sharing

With `share_mlp=True`, the decoder computes context metrics from the stack input and reuses them in every layer. With sharing disabled, each layer generates metrics from its own input. Generator parameters are registered at construction, so device conversion, optimizers and checkpoints include them before the first forward pass.

## Efficient RowSum

For the standard MLP, its final affine projection and RowSum commute. The code sums the corresponding output weights and biases before applying the final projection. This yields exactly the same effective head metrics and gradients while avoiding a full `[B, T, D, D]` causal tensor. The full output parameterization remains trainable.

## Code correspondence

| Paper component | Implementation |
| --- | --- |
| Input-dependent `fθ(X)`, Eq. (3) | `StructuredMLPForM.head_metrics` |
| Block-row sum, Eq. (4) | Fused output projection or explicit block-row sum |
| Per-head quadratic scores, Eqs. (5)–(6) | `MBasedAttention.forward` |
| Value aggregation and concatenation, Eqs. (7)–(8) | `MBasedAttention.forward` |
| Stack-wise sharing, Section 3.4 | `Decoder.forward` and `VisionTransformer.forward` |
| Static metric ablation | `use_M=True, use_mlp=False` |
| Independent generators | `use_M=True, use_mlp=True, share_mlp=False` |

## Ablations and API

The original `linear_comb`, `bayesian`, `separate_B`, compact-rank and layer-bias options remain available for experiments. Standard DAMCHA uses `off_diag_mode='mlp'`, scales of `1.0`, full rank, and no layer bias. `--use_multihead_M` is accepted for command compatibility; DAMCHA always uses RowSum per head.

Masks use `True` or integer `1` for visible entries. Floating-point attention masks add to the logits. `padding_mask` has shape `[B, T]` and marks real tokens. `get_M_matrices()` provides full metric matrices averaged across the last batch context for visualization; causal visualization uses the final prefix.
