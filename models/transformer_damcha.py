"""
DAMCHA Transformer: Data-Adaptive Mahalanobis Cross-Head Attention.

Key features:
- Decoder-only architecture for self-supervised masked reconstruction
- M-based attention: softmax(X @ M @ X.T / sqrt(d_head)) @ V
- M0 prior: Block matrix based on M^(h) = W_q^(h) @ W_k^(h).T structure
- MLP-based M generation with structural constraints
- Per-head metrics: sum cross-head blocks along each block row
"""

import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .m_attention import StructuredMLPForM, MBasedAttention


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


class DAMCHAAttention(MBasedAttention):
    """Tensor-output adapter for the decoder stack."""

    def __init__(self, d_model, n_heads, dropout=0.1, mlp_hidden=(256, 512),
                 dev_mode=False, use_M=False, use_mlp=False, layer_idx=0,
                 use_multihead_M=True, **kwargs):
        allowed = {'off_diag_mode', 'diag_scale', 'off_diag_scale',
                   'off_diag_alpha_init', 'use_layer_bias', 'layer_bias_rank',
                   'compact_rank', 'free_M', 'n_layers', 'shared_mlp'}
        options = {k: v for k, v in kwargs.items() if k in allowed}
        super().__init__(d_model, n_heads, dropout, use_M=use_M, use_mlp=use_mlp,
                         mlp_hidden_dims=mlp_hidden, layer_idx=layer_idx,
                         dev_mode=dev_mode, **options)

    def forward(self, x, attn_mask=None, head_metrics=None, padding_mask=None):
        return super().forward(x, attn_mask, is_causal=True,
                               padding_mask=padding_mask, head_metrics=head_metrics)[0]


class DecoderLayer(nn.Module):
    """Decoder layer with DAMCHA attention."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        dev_mode: bool = False,
        use_M: bool = False,
        use_mlp: bool = False,
        layer_idx: int = 0,
        use_multihead_M: bool = False,
        **kwargs
    ):
        super().__init__()
        self.dev_mode = dev_mode
        self.use_M = use_M
        self.use_mlp = use_mlp
        self.layer_idx = layer_idx
        self.use_multihead_M = use_multihead_M
        
        self.self_attn = DAMCHAAttention(
            d_model, n_heads, dropout, dev_mode=dev_mode,
            use_M=use_M, use_mlp=use_mlp,
            layer_idx=layer_idx, use_multihead_M=use_multihead_M, **kwargs
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def _causal_mask(self, T: int, device):
        """Create causal mask for autoregressive attention."""
        m = torch.full((T, T), float('-inf'), device=device)
        m = torch.triu(m, diagonal=1)
        return m
    
    def forward(self, x: torch.Tensor, head_metrics=None, padding_mask=None) -> torch.Tensor:
        B, T, _ = x.shape
        device = x.device
        
        if self.dev_mode:
            print(f"\n[DecoderLayer] Input shape: {x.shape}")
        
        attn_mask = self._causal_mask(T, device)
        sa = self.self_attn(x, head_metrics=head_metrics, padding_mask=padding_mask)
        x = self.ln1(x + self.dropout(sa))
        
        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))
        
        return x
    
    def get_M(self) -> torch.Tensor:
        """Get M from attention layer."""
        return self.self_attn.get_M()
    
    def get_M0(self) -> torch.Tensor:
        """Get M0 from attention layer."""
        return self.self_attn.get_M0()
    
    def get_attn(self) -> DAMCHAAttention:
        """Get the attention module."""
        return self.self_attn


class Decoder(nn.Module):
    """Decoder-only Transformer stack with DAMCHA attention."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float = 0.1,
        dev_mode: bool = False,
        use_M: bool = False,
        use_mlp: bool = False,
        share_mlp: bool = False,
        use_multihead_M: bool = False,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.dev_mode = dev_mode
        self.use_M = use_M
        self.use_mlp = use_mlp
        self.share_mlp = share_mlp and use_mlp and use_M
        self.use_multihead_M = use_multihead_M if use_M else False
        
        self.off_diag_mode = kwargs.get('off_diag_mode', 'mlp')
        self.use_layer_bias = kwargs.get('use_layer_bias', False)
        
        # Create shared MLP if enabled
        if self.share_mlp:
            mlp_hidden = kwargs.get('mlp_hidden', (256, 512))
            off_diag_scale = kwargs.get('off_diag_scale', 1.0)
            off_diag_alpha_init = kwargs.get('off_diag_alpha_init', 0.1)
            layer_bias_rank = kwargs.get('layer_bias_rank', 16)
            
            self.shared_mlp_module = StructuredMLPForM(
                d_model=d_model,
                n_heads=n_heads,
                hidden_dims=mlp_hidden,
                dropout=dropout,
                diag_scale=kwargs.get("diag_scale", 1.0),
                off_diag_mode=self.off_diag_mode,
                off_diag_scale=off_diag_scale,
                off_diag_alpha_init=off_diag_alpha_init,
                n_layers=n_layers,
                use_layer_bias=self.use_layer_bias,
                layer_bias_rank=layer_bias_rank,
                compact_rank=kwargs.get('compact_rank', 0),
                free_M=kwargs.get('free_M', False),
            )
            
            
            if dev_mode:
                print(f"[Decoder] Using SHARED MLP across {n_layers} blocks")
                print(f"  MLP params: {sum(p.numel() for p in self.shared_mlp_module.parameters()):,}")

        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model, n_heads, d_ff, dropout, dev_mode=dev_mode,
                use_M=use_M, use_mlp=use_mlp,
                layer_idx=i, use_multihead_M=use_multihead_M, n_layers=n_layers, shared_mlp=getattr(self, "shared_mlp_module", None), **kwargs
            )
            for i in range(n_layers)
        ])
        self.pe = PositionalEncoding(d_model)


    def forward(self, x: torch.Tensor, padding_mask=None) -> torch.Tensor:
        x = self.pe(x)
        metrics = None
        if self.share_mlp and not self.use_layer_bias:
            metrics = self.shared_mlp_module.head_metrics(x, causal=True, padding_mask=padding_mask)
        for layer in self.layers:
            x = layer(x, head_metrics=metrics, padding_mask=padding_mask)
        return x
    
    def get_M_matrices(self):
        """Get all M matrices from layers."""
        if not self.use_M:
            return []
        matrices = []
        for i, layer in enumerate(self.layers):
            M = layer.get_M()
            matrices.append((f'layer_{i}', M))
        return matrices
    
    def get_M0_matrices(self):
        """Get all M0 matrices from layers."""
        matrices = []
        for i, layer in enumerate(self.layers):
            M0 = layer.get_M0()
            matrices.append((f'layer_{i}', M0))
        return matrices
    
    def get_off_diag_params(self) -> dict:
        """Get off-diagonal parameters for saving."""
        if self.share_mlp and hasattr(self, 'shared_mlp_module'):
            return self.shared_mlp_module.get_off_diag_params()
        return {'off_diag_mode': self.off_diag_mode}
    
    def get_kl_divergence(self) -> torch.Tensor:
        """Get KL divergence for Bayesian mode."""
        if self.share_mlp and hasattr(self, 'shared_mlp_module'):
            return self.shared_mlp_module.get_kl_divergence()
        terms = [layer.get_attn().mlp_m.get_kl_divergence() for layer in self.layers
                 if layer.get_attn().use_mlp]
        return sum(terms, self.pe.pe.new_zeros(()))


class TransformerPC(nn.Module):
    """
    DAMCHA Transformer: Decoder-Only Transformer with M-based attention.
    
    Modes:
    - use_M=False: Standard Q/K/V Transformer attention (default)
    - use_M=True, use_mlp=False: M0-only attention
    - use_M=True, use_mlp=True: MLP-based M generation
    """
    
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        d_ff: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
        dev_mode: bool = False,
        use_M: bool = False,
        use_mlp: bool = False,
        share_mlp: bool = False,
        use_multihead_M: bool = False,
        **kwargs
    ):
        super().__init__()
        self.decoder = Decoder(
            d_model, n_heads, d_ff, n_layers, dropout,
            dev_mode=dev_mode, use_M=use_M, use_mlp=use_mlp,
            share_mlp=share_mlp, use_multihead_M=use_multihead_M, **kwargs
        )
        self.d_model = d_model
        self.dev_mode = dev_mode
        self.use_M = use_M
        self.use_mlp = use_mlp
        self.share_mlp = share_mlp
        self.use_multihead_M = use_multihead_M
    
    def forward(self, x: torch.Tensor, padding_mask=None) -> torch.Tensor:
        if self.dev_mode:
            mode = "standard" if not self.use_M else ("MLP" if self.use_mlp else "M0-only")
            print(f"\n[TransformerPC] Forward - Mode: {mode}, Training: {self.training}, Input: {x.shape}")
        
        return self.decoder(x, padding_mask=padding_mask)
    
    def get_M_matrices(self):
        """Get all M matrices from decoder layers."""
        return self.decoder.get_M_matrices()
    
    def get_M0_matrices(self):
        """Get all M0 matrices from decoder layers."""
        return self.decoder.get_M0_matrices()
    
    def get_off_diag_params(self) -> dict:
        """Get off-diagonal parameters."""
        return self.decoder.get_off_diag_params()
    
    def get_kl_divergence(self) -> torch.Tensor:
        """Get KL divergence for Bayesian mode."""
        return self.decoder.get_kl_divergence()
