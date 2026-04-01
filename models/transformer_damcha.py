"""
DAMCHA Transformer: Decoder-Only Transformer with Dynamic Attention M-matrix Cross-Head Aggregation.

Key features:
- Decoder-only architecture for self-supervised masked reconstruction
- M-based attention: softmax(X @ M @ X.T / sqrt(d_head)) @ V
- M0 prior: Block matrix based on M^(h) = W_q^(h) @ W_k^(h).T structure
- MLP-based M generation with structural constraints
- Multi-head M: aggregate off-diagonal blocks to diagonal, then extract per-head M
"""

import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .m_attention import StructuredMLPForM


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
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


class DAMCHAAttention(nn.Module):
    """
    DAMCHA: Dynamic Attention with M-matrix Cross-Head Aggregation.
    
    Attention mechanism: softmax(X @ M @ X.T / sqrt(d_head)) @ V
    
    M0 Prior Structure (FIXED, not updated):
    - M is a block matrix where M^(h) = W_q^(h) @ W_k^(h).T
    - Diagonal blocks: M^(h) for each head h
    - Off-diagonal blocks: M^(i,j) for cross-head interaction
    
    Modes:
    - use_M=False: Standard Q/K/V attention (default)
    - use_M=True, use_mlp=False: Use M0 matrix directly
    - use_M=True, use_mlp=True: MLP-based M generation
    
    Multi-head M mode (use_multihead_M=True):
    - Aggregate off-diagonal blocks to diagonal blocks
    - Extract per-head M matrices from aggregated diagonal blocks
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        mlp_hidden: Tuple[int, ...] = (256, 512),
        dev_mode: bool = False,
        use_M: bool = False,
        use_mlp: bool = False,
        layer_idx: int = 0,
        use_multihead_M: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dim_M = d_model * d_model
        self.dev_mode = dev_mode
        self.use_M = use_M
        self.use_mlp = use_mlp if use_M else False
        self.layer_idx = layer_idx
        self.use_multihead_M = use_multihead_M if use_M else False
        
        # Value projection and output projection
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Standard attention projections (used when use_M=False)
        if not use_M:
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
        
        # Initialize M0 prior
        self._init_M0_prior()
        
        # MLP for M generation
        self._mlp_config = {
            'hidden_dims': mlp_hidden,
            'dropout': dropout,
        }
        self._own_mlp = None
        self._shared_mlp = None
        
        # Cache for M
        self.register_buffer('cached_M', torch.zeros(self.dim_M), persistent=False)
    
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
    
    def get_M_from_mlp(self, layer_idx: int = 0) -> torch.Tensor:
        """Get M matrix from MLP."""
        return self.mlp_m.get_M_single(layer_idx=layer_idx)
    
    @torch.no_grad()
    def _init_M0_prior(self):
        """
        Initialize M0 prior based on standard attention structure.
        M^(h) = W_q^(h) @ W_k^(h).T for each head h
        """
        d, h, d_h = self.d_model, self.n_heads, self.d_head
        
        M0 = torch.zeros(d, d)
        
        # Initialize diagonal blocks
        for head in range(h):
            W_q = torch.empty(d_h, d_h)
            W_k = torch.empty(d_h, d_h)
            nn.init.orthogonal_(W_q)
            nn.init.orthogonal_(W_k)
            
            M_h = W_q @ W_k.t()
            start = head * d_h
            end = (head + 1) * d_h
            M0[start:end, start:end] = M_h
        
        # Initialize off-diagonal blocks
        for i in range(h):
            for j in range(h):
                if i != j:
                    start_i, end_i = i * d_h, (i + 1) * d_h
                    start_j, end_j = j * d_h, (j + 1) * d_h
                    M0[start_i:end_i, start_j:end_j] = 0.01 * torch.randn(d_h, d_h)
        
        self.register_buffer('M0', M0)
        
        if self.dev_mode:
            print(f"[DAMCHA] M0 initialized: shape={M0.shape}, norm={M0.norm():.4f}")
    
    def get_M(self) -> torch.Tensor:
        """Get cached M matrix."""
        return self.cached_M.view(self.d_model, self.d_model)
    
    def get_M0(self) -> torch.Tensor:
        """Get M0 prior matrix."""
        return self.M0
    
    def _aggregate_to_multihead(self, M: torch.Tensor) -> torch.Tensor:
        """
        Aggregate off-diagonal blocks to diagonal blocks, then extract per-head M.
        
        For each head h:
        M_h = (M_diag[h] + sum_{j!=h}(M[h,j] + M[j,h])) / (2*n_heads - 1)
        
        Args:
            M: Full M matrix [d_model, d_model]
            
        Returns:
            M_blocks: Per-head M matrices [n_heads, d_head, d_head]
        """
        M_blocks = []
        for h in range(self.n_heads):
            start_h, end_h = h * self.d_head, (h + 1) * self.d_head
            M_h = M[start_h:end_h, start_h:end_h].clone()
            
            # Add contributions from off-diagonal blocks
            for j in range(self.n_heads):
                if j != h:
                    start_j, end_j = j * self.d_head, (j + 1) * self.d_head
                    M_h = M_h + M[start_h:end_h, start_j:end_j]
                    M_h = M_h + M[start_j:end_j, start_h:end_h]
            
            # Normalize
            M_h = M_h / (2 * self.n_heads - 1)
            M_blocks.append(M_h)
        
        return torch.stack(M_blocks, dim=0)
    
    def _standard_attention(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard Q/K/V attention."""
        B, T, d = x.shape
        
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if attn_mask is not None:
            logits = logits + attn_mask.unsqueeze(0).unsqueeze(0)
        
        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, d)
        
        return self.out_proj(out)
    
    def _m_based_attention(
        self,
        x: torch.Tensor,
        M: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """M-based attention: softmax(X @ M @ X.T / sqrt(d_head)) @ V"""
        B, T, d = x.shape
        
        V = self.W_v(x)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # Expand M for batch if needed
        if M.dim() == 2:
            M_expanded = M.unsqueeze(0).expand(B, -1, -1)
        else:
            M_expanded = M
        
        if self.use_multihead_M:
            # Aggregate off-diagonal to diagonal, then extract per-head M
            M_blocks = self._aggregate_to_multihead(M)  # [nh, d_h, d_h]
            
            # Reshape x into heads
            x_heads = x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            
            # Per-head attention
            xM_heads = torch.einsum('bhtd,hde->bhte', x_heads, M_blocks)
            logits = torch.einsum('bhte,bhse->bhts', xM_heads, x_heads)
            logits = logits / math.sqrt(self.d_head)
            
            if self.dev_mode:
                print(f"[DAMCHA MultiHead] M_blocks: {M_blocks.shape}, logits: {logits.shape}")
        else:
            # Single M broadcast to all heads
            XM = torch.bmm(x, M_expanded)
            logits = torch.bmm(XM, x.transpose(1, 2)) / math.sqrt(self.d_head)
            logits = logits.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        
        if attn_mask is not None:
            logits = logits + attn_mask.unsqueeze(0).unsqueeze(0)
        
        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, d)
        
        return self.out_proj(out)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [B, T, d_model]
            attn_mask: Attention mask [T, T]
            
        Returns:
            Output [B, T, d_model]
        """
        if self.dev_mode:
            mode = "standard" if not self.use_M else ("MLP" if self.use_mlp else "M0-only")
            print(f"\n[DAMCHA] Forward - Mode: {mode}, Training: {self.training}, Input: {x.shape}")
        
        # Standard Q/K/V attention
        if not self.use_M:
            return self._standard_attention(x, attn_mask)
        
        # M0-only mode
        if not self.use_mlp:
            with torch.no_grad():
                self.cached_M = self.M0.flatten()
            return self._m_based_attention(x, self.M0, attn_mask)
        
        # MLP-based M generation
        M = self.get_M_from_mlp(layer_idx=self.layer_idx)
        with torch.no_grad():
            self.cached_M = M.flatten()
        
        return self._m_based_attention(x, M, attn_mask)


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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        device = x.device
        
        if self.dev_mode:
            print(f"\n[DecoderLayer] Input shape: {x.shape}")
        
        attn_mask = self._causal_mask(T, device)
        sa = self.self_attn(x, attn_mask=attn_mask)
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
        self.share_mlp = share_mlp and use_mlp
        self.use_multihead_M = use_multihead_M if use_M else False
        
        self.off_diag_mode = kwargs.get('off_diag_mode', 'mlp')
        self.use_layer_bias = kwargs.get('use_layer_bias', True)
        
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model, n_heads, d_ff, dropout, dev_mode=dev_mode,
                use_M=use_M, use_mlp=use_mlp,
                layer_idx=i, use_multihead_M=use_multihead_M, **kwargs
            )
            for i in range(n_layers)
        ])
        self.pe = PositionalEncoding(d_model)
        
        # Create shared MLP if enabled
        if self.share_mlp:
            mlp_hidden = kwargs.get('mlp_hidden', (256, 512))
            off_diag_scale = kwargs.get('off_diag_scale', 0.5)
            off_diag_alpha_init = kwargs.get('off_diag_alpha_init', 0.1)
            layer_bias_rank = kwargs.get('layer_bias_rank', 16)
            
            self.shared_mlp_module = StructuredMLPForM(
                d_model=d_model,
                n_heads=n_heads,
                hidden_dims=mlp_hidden,
                dropout=dropout,
                off_diag_mode=self.off_diag_mode,
                off_diag_scale=off_diag_scale,
                off_diag_alpha_init=off_diag_alpha_init,
                n_layers=n_layers,
                use_layer_bias=self.use_layer_bias,
                layer_bias_rank=layer_bias_rank,
                compact_rank=kwargs.get('compact_rank', 0),
                free_M=kwargs.get('free_M', False),
            )
            
            for layer in self.layers:
                layer.get_attn().set_shared_mlp(self.shared_mlp_module)
            
            if dev_mode:
                print(f"[Decoder] Using SHARED MLP across {n_layers} blocks")
                print(f"  MLP params: {sum(p.numel() for p in self.shared_mlp_module.parameters()):,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_M_matrices(self):
        """Get all M matrices from layers."""
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
        return torch.tensor(0.0)


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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dev_mode:
            mode = "standard" if not self.use_M else ("MLP" if self.use_mlp else "M0-only")
            print(f"\n[TransformerPC] Forward - Mode: {mode}, Training: {self.training}, Input: {x.shape}")
        
        return self.decoder(x)
    
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
