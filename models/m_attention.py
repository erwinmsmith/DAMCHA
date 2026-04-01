"""
M-based Attention Module.

This module provides M-based attention mechanism that can be integrated into
different model architectures (GPT, Diffusion, DAMCHA Transformer).

Key concept:
- Standard attention: softmax(Q @ K.T / sqrt(d)) @ V
- M-based attention: softmax(X @ M @ X.T / sqrt(d)) @ V

Where M is a learnable matrix that replaces the Q/K projection.

Modes:
- use_M=False: Standard Q/K/V attention
- use_M=True, use_mlp=False: Use M0 prior directly
- use_M=True, use_mlp=True: MLP generates M matrix
- share_mlp=True: Share MLP across all attention layers
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
        off_diag_scale: float = 0.5,
        off_diag_mode: str = 'mlp',
        off_diag_alpha_init: float = 0.1,
        n_layers: int = 4,
        use_layer_bias: bool = True,
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
        
        self.input_embedding = nn.Parameter(torch.randn(d_model))
        self.diag_block_weights = nn.Parameter(torch.ones(n_heads))
        
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
        else:
            # Legacy: simple off-diagonal weights (not used in linear_comb mode)
            self.off_diag_weights = nn.Parameter(torch.ones(n_heads, n_heads) * 0.1)
        
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
    
    def forward(self, batch_size: int = 1, layer_idx: int = 0) -> torch.Tensor:
        """
        Generate structured M matrix.
        
        Args:
            batch_size: Number of M matrices to generate
            layer_idx: Index of the transformer layer (for per-layer bias)
            
        Returns:
            M matrix [batch_size, d_model, d_model] with block structure
        """
        x = self.input_embedding.unsqueeze(0).expand(batch_size, -1)
        M_flat = self.mlp(x)

        if self.compact_rank > 0:
            # Low-rank compact path: M = U @ V^T  (full D×D, low-rank factored)
            D, r = self.d_model, self.compact_rank
            UV = M_flat.view(batch_size, 2, D, r)
            U, V = UV[:, 0], UV[:, 1]                      # [B, D, r] each
            M = torch.bmm(U, V.transpose(-1, -2)) / (r ** 0.5)  # [B, D, D]
        else:
            # Original full-rank path
            M_raw = M_flat.view(batch_size, self.d_model, self.d_model)
            if self.free_M:
                # Ablation: unconstrained D×D M, no block masking applied
                M = M_raw
            else:
                M_diag = M_raw * self.diag_mask.unsqueeze(0) * self.diag_scale
                if self.off_diag_mode == 'mlp':
                    M_off_diag = M_raw * self.off_diag_mask.unsqueeze(0) * self.off_diag_scale
                elif self.off_diag_mode == 'separate_B':
                    # Each block-row has its own B_i [d_h, D]; assemble into [D, D] then mask
                    B_full = self.B.reshape(self.d_model, self.d_model)  # [H*d_h, D] = [D, D]
                    M_off_diag = B_full.unsqueeze(0) * self.off_diag_mask.unsqueeze(0) * self.off_diag_scale
                else:
                    diag_blocks = self._extract_diagonal_blocks(M_diag)
                    sample_bayesian = self.training and self.off_diag_mode == 'bayesian'
                    M_off_diag = self._compute_off_diag_from_linear_comb(diag_blocks, sample_bayesian)
                    M_off_diag = M_off_diag * self.off_diag_scale
                M = M_diag + M_off_diag

        # Add per-layer bias if enabled (low-rank factorization)
        if self.use_layer_bias:
            layer_idx = min(layer_idx, self.n_layers - 1)  # Clamp to valid range
            layer_bias = self._get_layer_bias(layer_idx)
            M = M + layer_bias.unsqueeze(0)

        return M
    
    def get_M_single(self, layer_idx: int = 0) -> torch.Tensor:
        """
        Get single M matrix (no batch dimension).
        
        Args:
            layer_idx: Index of the transformer layer (for per-layer bias)
        """
        return self.forward(batch_size=1, layer_idx=layer_idx).squeeze(0)
    
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
            return torch.tensor(0.0, device=self.off_diag_alpha.device)
        
        # KL divergence for diagonal Gaussian
        # KL = 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))
        mu = self.off_diag_alpha
        logvar = self.off_diag_alpha_logvar
        
        # Only compute for off-diagonal blocks (i != j)
        h = self.n_heads
        kl = 0.0
        for i in range(h):
            for j in range(h):
                if i != j:
                    mu_ij = mu[i, j]  # [h]
                    logvar_ij = logvar[i, j]  # [h]
                    kl += 0.5 * torch.sum(torch.exp(logvar_ij) + mu_ij**2 - 1 - logvar_ij)
        
        return kl


class MBasedAttention(nn.Module):
    """
    M-based Attention Module.
    
    Can operate in multiple modes:
    - Standard Q/K/V attention (use_M=False)
    - M0 prior attention (use_M=True, use_mlp=False)
    - MLP-generated M attention (use_M=True, use_mlp=True)
    
    This module is designed to be integrated into different architectures.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        use_M: bool = False,
        use_mlp: bool = False,
        mlp_hidden_dims: Tuple[int, ...] = (256, 512),
        dev_mode: bool = False,
        use_multihead_M: bool = False,
    ):
        """
        Initialize M-based attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            bias: Whether to use bias in linear layers
            use_M: Use M-based attention instead of Q/K
            use_mlp: Use MLP to generate M (requires use_M=True)
            mlp_hidden_dims: Hidden dimensions for MLP
            dev_mode: Enable debug output
            use_multihead_M: If True, row-normalize M and extract h diagonal blocks,
                             each block serves as M_h for head h (true multi-head M attention)
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        self.use_M = use_M
        self.use_mlp = use_mlp if use_M else False
        self.dev_mode = dev_mode
        self.use_multihead_M = use_multihead_M if use_M else False
        
        # Value projection (always needed)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        if not use_M:
            # Standard Q/K projections
            self.W_q = nn.Linear(d_model, d_model, bias=bias)
            self.W_k = nn.Linear(d_model, d_model, bias=bias)
        else:
            # Initialize M0 prior
            self._init_M0_prior()
            
            # MLP for M generation (created lazily or set externally)
            self._mlp_config = {
                'hidden_dims': mlp_hidden_dims,
                'dropout': dropout,
            }
            self._own_mlp = None
            self._shared_mlp = None
    
    @torch.no_grad()
    def _init_M0_prior(self):
        """
        Initialize M0 prior based on standard attention structure.
        M^(h) = W_q^(h) @ W_k^(h).T for each head h
        """
        d, h, d_h = self.d_model, self.n_heads, self.d_head
        
        M0 = torch.zeros(d, d)
        
        for head in range(h):
            W_q = torch.empty(d_h, d_h)
            W_k = torch.empty(d_h, d_h)
            nn.init.orthogonal_(W_q)
            nn.init.orthogonal_(W_k)
            
            M_h = W_q @ W_k.t()
            
            start = head * d_h
            end = (head + 1) * d_h
            M0[start:end, start:end] = M_h
        
        for i in range(h):
            for j in range(h):
                if i != j:
                    start_i, end_i = i * d_h, (i + 1) * d_h
                    start_j, end_j = j * d_h, (j + 1) * d_h
                    M0[start_i:end_i, start_j:end_j] = 0.01 * torch.randn(d_h, d_h)
        
        self.register_buffer('M0', M0)
        
        if self.dev_mode:
            print(f"[M-Attn] M0 prior initialized: shape={M0.shape}, norm={M0.norm():.4f}")
    
    def _create_own_mlp(self):
        """Lazily create own MLP if not using shared."""
        if self._own_mlp is None:
            self._own_mlp = StructuredMLPForM(
                d_model=self.d_model,
                n_heads=self.n_heads,
                hidden_dims=self._mlp_config['hidden_dims'],
                dropout=self._mlp_config['dropout'],
            )
        return self._own_mlp
    
    @property
    def mlp_m(self) -> StructuredMLPForM:
        """Get the active MLP for M generation (shared or own)."""
        if self._shared_mlp is not None:
            return self._shared_mlp
        return self._create_own_mlp()
    
    def set_shared_mlp(self, shared_mlp: StructuredMLPForM):
        """Set a shared MLP to be used instead of the own MLP."""
        self._shared_mlp = shared_mlp
        self._own_mlp = None
    
    def use_own_mlp(self):
        """Revert to using own MLP instead of shared."""
        self._shared_mlp = None
    
    def get_M(self) -> torch.Tensor:
        """Get the M matrix (from MLP or M0)."""
        if self.use_mlp:
            return self.mlp_m.get_M_single()
        else:
            return self.M0
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for M-based attention.
        
        Args:
            x: Input tensor [B, T, d_model]
            attn_mask: Optional attention mask [T, T] or [B, T, T]
            is_causal: Whether to apply causal masking
            
        Returns:
            Tuple of (output, attention_weights)
            attention_weights is None if using flash attention
        """
        B, T, d = x.shape
        
        # Value projection
        v = self.W_v(x)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, nh, T, d_h]
        
        if not self.use_M:
            # Standard Q/K attention
            q = self.W_q(x)
            k = self.W_k(x)
            q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            
            # Attention scores
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        else:
            # M-based attention: X @ M @ X.T
            M = self.get_M()  # [d_model, d_model]
            
            if self.use_multihead_M:
                # True multi-head M attention:
                # 1. Aggregate off-diagonal blocks to diagonal blocks
                # 2. Extract h diagonal blocks (each d_head x d_head)
                # 3. Each block serves as M_h for head h
                
                # For each diagonal block h, add contributions from off-diagonal blocks
                # M_h_aggregated = M_diag[h] + sum_{j!=h}(M_off[h,j] + M_off[j,h]) / (2*(n_heads-1))
                M_blocks = []
                for h in range(self.n_heads):
                    # Start with diagonal block
                    start_h, end_h = h * self.d_head, (h + 1) * self.d_head
                    M_h = M[start_h:end_h, start_h:end_h].clone()
                    
                    # Add contributions from off-diagonal blocks
                    for j in range(self.n_heads):
                        if j != h:
                            start_j, end_j = j * self.d_head, (j + 1) * self.d_head
                            # Add M[h,j] (row h, col j) and M[j,h] (row j, col h)
                            M_h = M_h + M[start_h:end_h, start_j:end_j]
                            M_h = M_h + M[start_j:end_j, start_h:end_h]
                    
                    # Normalize by number of added blocks (2*(n_heads-1) off-diag + 1 diag)
                    M_h = M_h / (2 * self.n_heads - 1)
                    M_blocks.append(M_h)
                
                M_blocks = torch.stack(M_blocks, dim=0)  # [nh, d_h, d_h]
                
                # Reshape x into heads: [B, T, d] -> [B, T, nh, d_h] -> [B, nh, T, d_h]
                x_heads = x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, nh, T, d_h]
                
                # Per-head attention: x_h @ M_h @ x_h.T for each head
                # x_heads: [B, nh, T, d_h]
                # M_blocks: [nh, d_h, d_h]
                # xM_h = x_h @ M_h: [B, nh, T, d_h]
                xM_heads = torch.einsum('bhtd,hde->bhte', x_heads, M_blocks)  # [B, nh, T, d_h]
                
                # att_h = xM_h @ x_h.T: [B, nh, T, T]
                att = torch.einsum('bhte,bhse->bhts', xM_heads, x_heads)  # [B, nh, T, T]
                att = att * (1.0 / math.sqrt(self.d_head))
                
                if self.dev_mode:
                    print(f"[M-Attn MultiHead] M_blocks: {M_blocks.shape}, x_heads: {x_heads.shape}")
            else:
                # Original single-head M broadcast to all heads
                # Compute X @ M @ X.T
                xM = x @ M  # [B, T, d_model]
                att_full = xM @ x.transpose(-2, -1)  # [B, T, T]
                
                # Reshape for multi-head: split into heads
                # att_full is [B, T, T], we need [B, nh, T, T]
                # For M-based attention, we compute per-head attention from the full matrix
                att = att_full.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
                att = att * (1.0 / math.sqrt(self.d_head))
        
        # Apply causal mask if needed
        if is_causal:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            att = att.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            att = att.masked_fill(attn_mask == 0, float('-inf'))
        
        # Softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = att @ v  # [B, nh, T, d_h]
        
        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, d)
        y = self.resid_dropout(self.out_proj(y))
        
        if self.dev_mode:
            print(f"[M-Attn] x: {x.shape}, att: {att.shape}, y: {y.shape}")
            if self.use_M:
                print(f"[M-Attn] M: {M.shape}, M norm: {M.norm():.4f}")
        
        return y, att if not self.use_M else None
    
    def get_intermediate_matrices(self) -> dict:
        """
        Get intermediate matrices for debugging/analysis.
        
        Returns:
            Dictionary containing M0, M (if use_mlp), etc.
        """
        result = {}
        if self.use_M:
            result['M0'] = self.M0.clone()
            if self.use_mlp:
                result['M'] = self.get_M().clone()
        return result


def create_shared_mlp(
    d_model: int,
    n_heads: int,
    hidden_dims: Tuple[int, ...] = (256, 512),
    dropout: float = 0.1,
    off_diag_mode: str = 'mlp',
) -> StructuredMLPForM:
    """
    Create a shared MLP for M generation.
    
    This MLP can be shared across multiple attention layers.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        hidden_dims: Hidden layer dimensions
        dropout: Dropout rate
        off_diag_mode: 'mlp' | 'linear_comb' | 'bayesian'
        
    Returns:
        StructuredMLPForM instance
    """
    return StructuredMLPForM(
        d_model=d_model,
        n_heads=n_heads,
        hidden_dims=hidden_dims,
        dropout=dropout,
        off_diag_mode=off_diag_mode,
    )


def setup_shared_mlp_for_layers(
    attention_layers: list,
    d_model: int,
    n_heads: int,
    hidden_dims: Tuple[int, ...] = (256, 512),
    dropout: float = 0.1,
    off_diag_mode: str = 'mlp',
) -> StructuredMLPForM:
    """
    Create and set up a shared MLP for multiple attention layers.
    
    Args:
        attention_layers: List of MBasedAttention layers
        d_model: Model dimension
        n_heads: Number of attention heads
        hidden_dims: Hidden layer dimensions
        dropout: Dropout rate
        off_diag_mode: 'mlp' | 'linear_comb' | 'bayesian'
        
    Returns:
        The shared MLP instance
    """
    shared_mlp = create_shared_mlp(d_model, n_heads, hidden_dims, dropout, off_diag_mode)
    
    for layer in attention_layers:
        if hasattr(layer, 'set_shared_mlp'):
            layer.set_shared_mlp(shared_mlp)
    
    return shared_mlp
