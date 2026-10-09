"""
Data-Adaptive Mahalanobis Cross-Head Attention.

This module provides M-based attention mechanism that can be integrated into
different model architectures (GPT, Diffusion, DAMCHA Transformer).

M(X) is generated from a mean-pooled input context. Head i uses the sum
of the blocks in row i (paper Eqs. 3--8 and Appendix D). Autoregressive
queries use prefix means; noncausal attention uses the whole sequence.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuredMLPForM(nn.Module):
    """
    MLP for generating structured M matrix with block-diagonal constraints.
    
    The M matrix maintains a block structure:
    - Diagonal blocks: Strong learned patterns (per-head attention)
    - Off-diagonal blocks: Can be generated via MLP or as linear combination of diagonal blocks
    
    Modes:
    - off_diag_mode='mlp': Off-diagonal blocks generated directly by MLP (default)
    - off_diag_mode='linear_comb': Off-diagonal block (i,j) = sum_k(alpha_{i,j,k} * diag_block_k)
    - off_diag_mode='bayesian': Same as linear_comb but alpha is sampled from learned distribution
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        hidden_dims: Tuple[int, ...] = (256, 512),
        dropout: float = 0.1,
        diag_scale: float = 1.0,
        off_diag_scale: float = 1.0,
        off_diag_mode: str = 'mlp',
        off_diag_alpha_init: float = 0.1,
        n_layers: int = 4,
        use_layer_bias: bool = False,
        layer_bias_rank: int = 16,
        compact_rank: int = 0,
        free_M: bool = False,
    ):
        """
        Initialize structured MLP for M matrix generation.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            diag_scale: Scale factor for diagonal blocks
            off_diag_scale: Scale factor for off-diagonal blocks
            off_diag_mode: 'mlp' | 'linear_comb' | 'bayesian'
            off_diag_alpha_init: Initial std for off-diagonal alpha weights
            n_layers: Number of transformer layers (for layer-specific bias)
            use_layer_bias: Whether to use learnable per-layer bias for M
            layer_bias_rank: Rank for low-rank layer bias factorization
            free_M: If True, skip block masking and use raw D×D MLP output (ablation: no structural constraint)
        """
        super().__init__()
        if n_heads <= 0 or d_model <= 0 or d_model % n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if off_diag_mode not in {"mlp", "linear_comb", "bayesian", "separate_B"}:
            raise ValueError("Unknown off_diag_mode")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.compact_rank = compact_rank
        self.free_M = free_M
        self.dim_M = 2 * d_model * compact_rank if compact_rank > 0 else d_model * d_model
        self.diag_scale = diag_scale
        self.off_diag_scale = off_diag_scale
        self.off_diag_mode = off_diag_mode
        self.off_diag_alpha_init = off_diag_alpha_init
        self.n_layers = n_layers
        self.use_layer_bias = use_layer_bias
        
        layers = []
        input_dim = d_model
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, self.dim_M))
        self.mlp = nn.Sequential(*layers)
        
        
        # Off-diagonal combination weights
        # For each off-diagonal block (i, j) where i != j, we have n_heads combination coefficients
        # Shape: [n_heads, n_heads, n_heads] -> alpha[i, j, k] is the weight of diagonal block k
        # for constructing off-diagonal block (i, j)
        n_off_diag = n_heads * (n_heads - 1)  # Number of off-diagonal blocks
        
        if off_diag_mode in ['linear_comb', 'bayesian']:
            # Combination weights: alpha[i, j, k] for off-diagonal block (i,j) using diagonal block k
            # Initialize with small values, slight preference for nearby heads
            self.off_diag_alpha = nn.Parameter(torch.zeros(n_heads, n_heads, n_heads))
            self._init_off_diag_alpha()
            
            if off_diag_mode == 'bayesian':
                # Log variance for each combination weight (for reparameterization trick)
                # Initialize to small variance (log(0.01) ~ -4.6)
                self.off_diag_alpha_logvar = nn.Parameter(torch.full((n_heads, n_heads, n_heads), -4.0))
        elif off_diag_mode == 'separate_B':
            # One independent B matrix per block-row (head): shape [H, d_h, D]
            # B[i] controls the entire i-th block row's off-diagonal entries
            # After reshape to [D, D] and masking, diagonal blocks are zeroed out
            self.B = nn.Parameter(torch.zeros(n_heads, d_model // n_heads, d_model))
            nn.init.normal_(self.B, mean=0.0, std=0.01)
        self._init_structural_bias()
        
        # Learnable per-layer bias for M matrix using low-rank factorization
        # layer_bias_i = U @ diag(layer_scales[i]) @ V.T
        # This shares U, V across layers, only layer_scales differs per layer
        self.use_layer_bias = use_layer_bias
        self.layer_bias_rank = layer_bias_rank
        if use_layer_bias:
            # Shared basis matrices
            self.layer_bias_U = nn.Parameter(torch.zeros(d_model, layer_bias_rank))
            self.layer_bias_V = nn.Parameter(torch.zeros(d_model, layer_bias_rank))
            # Per-layer scaling vectors
            self.layer_scales = nn.Parameter(torch.zeros(n_layers, layer_bias_rank))
            self._init_layer_bias()
    
    def _init_layer_bias(self):
        """Initialize low-rank layer bias with small values."""
        with torch.no_grad():
            # Initialize U, V with small orthogonal-like values
            nn.init.orthogonal_(self.layer_bias_U, gain=0.1)
            nn.init.orthogonal_(self.layer_bias_V, gain=0.1)
            # Initialize per-layer scales with layer-dependent values
            for layer_idx in range(self.n_layers):
                scale = 0.1 * (1 + layer_idx * 0.2)
                self.layer_scales.data[layer_idx] = torch.randn(self.layer_bias_rank) * scale
    
    def _get_layer_bias(self, layer_idx: int) -> torch.Tensor:
        """Compute layer bias from low-rank factorization.
        
        layer_bias = U @ diag(layer_scales[layer_idx]) @ V.T
        """
        scales = self.layer_scales[layer_idx]  # [rank]
        # U @ diag(s) @ V.T = (U * s) @ V.T
        return (self.layer_bias_U * scales.unsqueeze(0)) @ self.layer_bias_V.T
    
    def _init_off_diag_alpha(self):
        """Initialize off-diagonal combination weights.
        
        Use small random initialization so the network learns which 
        off-diagonal blocks should be significant through training.
        No fixed pattern based on distance - let it emerge naturally.
        """
        h = self.n_heads
        with torch.no_grad():
            # Small random initialization for all alpha values
            # The network will learn to increase weights for important blocks
            self.off_diag_alpha.data.normal_(0, self.off_diag_alpha_init)
    
    def _init_structural_bias(self):
        """Initialize structural bias based on M0 pattern."""
        d, h, d_h = self.d_model, self.n_heads, self.d_head
        
        diag_mask = torch.zeros(d, d)
        off_diag_mask = torch.zeros(d, d)
        
        for head in range(h):
            start = head * d_h
            end = (head + 1) * d_h
            diag_mask[start:end, start:end] = 1.0
        
        off_diag_mask = 1.0 - diag_mask
        
        self.register_buffer('diag_mask', diag_mask)
        self.register_buffer('off_diag_mask', off_diag_mask)
    
    def _extract_diagonal_blocks(self, M: torch.Tensor) -> torch.Tensor:
        """
        Extract diagonal blocks from M matrix.
        
        Args:
            M: Full M matrix [batch_size, d_model, d_model]
            
        Returns:
            Diagonal blocks [batch_size, n_heads, d_head, d_head]
        """
        B = M.shape[0]
        h, d_h = self.n_heads, self.d_head
        
        diag_blocks = torch.zeros(B, h, d_h, d_h, device=M.device, dtype=M.dtype)
        for k in range(h):
            start = k * d_h
            end = (k + 1) * d_h
            diag_blocks[:, k] = M[:, start:end, start:end]
        
        return diag_blocks
    
    def _compute_off_diag_from_linear_comb(
        self,
        diag_blocks: torch.Tensor,
        sample_bayesian: bool = False
    ) -> torch.Tensor:
        """
        Compute off-diagonal blocks as linear combination of diagonal blocks.
        
        Off-diagonal block (i, j) = sum_k(alpha[i, j, k] * diag_block[k])
        
        Args:
            diag_blocks: Diagonal blocks [batch_size, n_heads, d_head, d_head]
            sample_bayesian: If True, sample alpha from distribution (for bayesian mode)
            
        Returns:
            Off-diagonal contribution to M [batch_size, d_model, d_model]
        """
        B = diag_blocks.shape[0]
        h, d_h = self.n_heads, self.d_head
        d = self.d_model
        
        # Get combination weights
        if sample_bayesian and self.off_diag_mode == 'bayesian':
            # Reparameterization trick: alpha = mu + sigma * epsilon
            std = torch.exp(0.5 * self.off_diag_alpha_logvar)
            eps = torch.randn_like(std)
            alpha = self.off_diag_alpha + std * eps
        else:
            alpha = self.off_diag_alpha
        
        # Construct off-diagonal blocks
        M_off_diag = torch.zeros(B, d, d, device=diag_blocks.device, dtype=diag_blocks.dtype)
        
        for i in range(h):
            for j in range(h):
                if i != j:
                    # Compute linear combination: sum_k(alpha[i,j,k] * diag_block[k])
                    # diag_blocks: [B, h, d_h, d_h]
                    # alpha[i, j]: [h] -> weights for each diagonal block k
                    weights = alpha[i, j]  # [h]
                    
                    # Weighted sum of diagonal blocks
                    # [B, k, d_h, d_h] * [k] -> sum over k -> [B, d_h, d_h]
                    # einsum: 'bkhw,k->bhw' means sum over k dimension
                    off_block = torch.einsum('bkij,k->bij', diag_blocks, weights)
                    
                    # Place in off-diagonal position (i, j)
                    start_i, end_i = i * d_h, (i + 1) * d_h
                    start_j, end_j = j * d_h, (j + 1) * d_h
                    M_off_diag[:, start_i:end_i, start_j:end_j] = off_block
        
        return M_off_diag
    
    def forward(self, context: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """
        Generate structured M matrix.
        
        Args:
            context: Pooled input features [batch, d_model]
            layer_idx: Index of the transformer layer (for per-layer bias)
            
        Returns:
            M matrix [batch_size, d_model, d_model] with block structure
        """
        batch_size = context.shape[0]
        M_flat = self.mlp(context)

        if self.compact_rank > 0:
            d, rank = self.d_model, self.compact_rank
            factors = M_flat.view(batch_size, 2, d, rank)
            M = torch.bmm(factors[:, 0], factors[:, 1].transpose(-1, -2)) / math.sqrt(rank)
        else:
            raw = M_flat.view(batch_size, self.d_model, self.d_model)
            if self.free_M or self.off_diag_mode == 'mlp':
                M = raw
            else:
                diagonal = raw * self.diag_mask
                if self.off_diag_mode == 'separate_B':
                    off_diagonal = self.B.reshape(self.d_model, self.d_model) * self.off_diag_mask
                else:
                    blocks = self._extract_diagonal_blocks(raw)
                    off_diagonal = self._compute_off_diag_from_linear_comb(
                        blocks, self.training and self.off_diag_mode == 'bayesian')
                M = diagonal + off_diagonal
        if self.use_layer_bias:
            if not 0 <= layer_idx < self.n_layers:
                raise ValueError('layer_idx is out of range')
            M = M + self._get_layer_bias(layer_idx)
        # Scale after generation so block ablations also cover biases and compact metrics.
        M = M * (self.diag_mask * self.diag_scale + self.off_diag_mask * self.off_diag_scale)

        return M
    
    def head_metrics(self, x, causal=False, padding_mask=None, layer_idx=0):
        """Mean-conditioned row sums (Eq. 4); prefix means for causal queries.

        Fuse RowSum into the last linear layer in the unconstrained MLP path.
        This is algebraically exact and avoids storing a D x D matrix per token.
        """
        valid = torch.ones_like(x[..., :1]) if padding_mask is None else padding_mask[..., None].to(x)
        if causal:
            context = (x * valid).cumsum(1) / valid.cumsum(1).clamp_min(1)
        else:
            context = (x * valid).sum(1) / valid.sum(1).clamp_min(1)
        self.last_context = (context[:, -1] if causal else context).detach()
        h, dh, d = self.n_heads, self.d_head, self.d_model
        if self.compact_rank == 0 and self.off_diag_mode == 'mlp' and not self.use_layer_bias:
            hidden = self.mlp[:-1](context)
            last = self.mlp[-1]
            scale = (self.diag_mask * self.diag_scale + self.off_diag_mask * self.off_diag_scale)
            weight = (last.weight * scale.flatten()[:, None]).reshape(h, dh, h, dh, -1).sum(2)
            bias = (last.bias * scale.flatten()).reshape(h, dh, h, dh).sum(2)
            return F.linear(hidden, weight.flatten(0, 2), bias.flatten()).reshape(*context.shape[:-1], h, dh, dh)
        full = self(context.reshape(-1, d), layer_idx=layer_idx)
        rows = full.reshape(-1, h, dh, h, dh).sum(3)
        return rows.reshape(*context.shape[:-1], h, dh, dh)

    def get_off_diag_params(self) -> dict:
        """
        Get off-diagonal combination parameters for saving.
        
        Returns:
            Dictionary containing off-diagonal parameters
        """
        params = {
            'off_diag_mode': self.off_diag_mode,
            'off_diag_scale': self.off_diag_scale,
            'use_layer_bias': self.use_layer_bias,
            'n_layers': self.n_layers,
        }
        
        if self.off_diag_mode in ['linear_comb', 'bayesian']:
            params['off_diag_alpha'] = self.off_diag_alpha.detach().cpu()
            if self.off_diag_mode == 'bayesian':
                params['off_diag_alpha_logvar'] = self.off_diag_alpha_logvar.detach().cpu()
                # Also compute and save the standard deviation
                params['off_diag_alpha_std'] = torch.exp(0.5 * self.off_diag_alpha_logvar).detach().cpu()
        
        # Save per-layer bias if enabled (low-rank factorization)
        if self.use_layer_bias:
            params['layer_bias_rank'] = self.layer_bias_rank
            params['layer_bias_U'] = self.layer_bias_U.detach().cpu()
            params['layer_bias_V'] = self.layer_bias_V.detach().cpu()
            params['layer_scales'] = self.layer_scales.detach().cpu()
        
        return params
    
    def get_kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence for Bayesian mode (for ELBO loss).
        
        KL(q(alpha) || p(alpha)) where:
        - q(alpha) = N(mu, sigma^2) is the learned posterior
        - p(alpha) = N(0, 1) is the prior
        
        Returns:
            KL divergence scalar
        """
        if self.off_diag_mode != 'bayesian':
            return self.diag_mask.new_zeros(())
        
        # KL divergence for diagonal Gaussian
        # KL = 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))
        mu = self.off_diag_alpha
        logvar = self.off_diag_alpha_logvar
        
        # Only compute for off-diagonal blocks (i != j)
        h = self.n_heads
        kl = mu.new_zeros(())
        for i in range(h):
            for j in range(h):
                if i != j:
                    mu_ij = mu[i, j]  # [h]
                    logvar_ij = logvar[i, j]  # [h]
                    kl += 0.5 * torch.sum(torch.exp(logvar_ij) + mu_ij**2 - 1 - logvar_ij)
        
        return kl


class MBasedAttention(nn.Module):
    """Input-adaptive cross-head attention, paper Eqs. (3)--(8).

    Boolean/integer masks use True/1 for visible keys; floating masks are
    additive logits. padding_mask is [B,T] with True for real tokens.
    """

    def __init__(self, d_model, n_heads, dropout=0.1, bias=True, use_M=False,
                 use_mlp=False, mlp_hidden_dims=(256, 512), dev_mode=False,
                 use_multihead_M=True, layer_idx=0, shared_mlp=None, **metric_options):
        super().__init__()
        if n_heads <= 0 or d_model % n_heads:
            raise ValueError('d_model must be divisible by positive n_heads')
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.use_M, self.use_mlp = use_M, use_M and use_mlp
        self.use_multihead_M = use_M
        self.dev_mode, self.layer_idx = dev_mode, layer_idx
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self._shared_mlp = shared_mlp
        self._own_mlp = None
        self.register_buffer('cached_M', torch.zeros(d_model, d_model), persistent=False)
        if use_M:
            self.M0 = nn.Parameter(torch.eye(d_model)) if not use_mlp else None
            if use_mlp and shared_mlp is None:
                self._own_mlp = StructuredMLPForM(d_model, n_heads,
                    hidden_dims=mlp_hidden_dims, dropout=dropout, **metric_options)
        else:
            self.W_q = nn.Linear(d_model, d_model, bias=bias)
            self.W_k = nn.Linear(d_model, d_model, bias=bias)

    @property
    def mlp_m(self):
        return self._shared_mlp if self._shared_mlp is not None else self._own_mlp

    def set_shared_mlp(self, shared_mlp):
        self._shared_mlp = shared_mlp
        self._own_mlp = None

    def use_own_mlp(self):
        import copy
        if self._shared_mlp is not None:
            self._own_mlp = copy.deepcopy(self._shared_mlp)
            self._shared_mlp = None

    @torch.no_grad()
    def get_M(self):
        if self.use_mlp and hasattr(self.mlp_m, 'last_context'):
            return self.mlp_m(self.mlp_m.last_context, layer_idx=self.layer_idx).mean(0)
        return self.M0 if self.use_M and not self.use_mlp else self.cached_M

    def get_M0(self):
        return self.M0 if getattr(self, 'M0', None) is not None else self.cached_M.new_zeros(self.d_model, self.d_model)

    def _aggregate_to_multihead(self, M):
        return M.reshape(*M.shape[:-2], self.n_heads, self.d_head,
                         self.n_heads, self.d_head).sum(-2)

    def forward(self, x, attn_mask=None, is_causal=False, padding_mask=None,
                head_metrics=None):
        b, t, d = x.shape
        v = self.W_v(x).reshape(b, t, self.n_heads, self.d_head).transpose(1, 2)
        if self.use_M:
            xh = x.reshape(b, t, self.n_heads, self.d_head).transpose(1, 2)
            if head_metrics is None:
                head_metrics = (self.mlp_m.head_metrics(x, is_causal, padding_mask, self.layer_idx)
                                if self.use_mlp else self._aggregate_to_multihead(self.M0))
            if head_metrics.ndim == 5:
                q = torch.einsum('bhtd,bthde->bhte', xh, head_metrics)
            else:
                q = torch.matmul(xh, head_metrics)
            scores = q @ xh.transpose(-2, -1) / math.sqrt(self.d_head)
        else:
            q = self.W_q(x).reshape(b, t, self.n_heads, self.d_head).transpose(1, 2)
            k = self.W_k(x).reshape(b, t, self.n_heads, self.d_head).transpose(1, 2)
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        if is_causal:
            future = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(future, float('-inf'))
        if padding_mask is not None:
            scores = scores.masked_fill(~padding_mask[:, None, None, :].bool(), float('-inf'))
        if attn_mask is not None:
            if attn_mask.ndim == 3:
                attn_mask = attn_mask[:, None]
            if attn_mask.dtype == torch.bool or not attn_mask.is_floating_point():
                scores = scores.masked_fill(~attn_mask.bool(), float('-inf'))
            else:
                scores = scores + attn_mask
        # A fully padded query has zero attention and zero contribution.
        all_masked = torch.isneginf(scores).all(-1, keepdim=True)
        weights = F.softmax(scores.masked_fill(all_masked, 0), dim=-1).masked_fill(all_masked, 0)
        weights = self.attn_dropout(weights)
        y = (weights @ v).transpose(1, 2).reshape(b, t, d)
        y = self.resid_dropout(self.out_proj(y))
        if padding_mask is not None:
            y = y * padding_mask[..., None].to(y)
        return y, weights


def create_shared_mlp(d_model, n_heads, hidden_dims=(256, 512), dropout=0.1,
                      off_diag_mode='mlp', **kwargs):
    return StructuredMLPForM(d_model, n_heads, hidden_dims, dropout,
                            off_diag_mode=off_diag_mode, **kwargs)


def setup_shared_mlp_for_layers(attention_layers, d_model, n_heads,
                                hidden_dims=(256, 512), dropout=0.1, off_diag_mode='mlp'):
    shared = create_shared_mlp(d_model, n_heads, hidden_dims, dropout, off_diag_mode)
    for layer in attention_layers:
        layer.set_shared_mlp(shared)
    return shared
