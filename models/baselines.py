"""Baseline attention mechanisms for comparison with DAMCHA (M-attention).

Baselines:
  1. THA    - Talking-Heads Attention (Shazeer et al., 2020)
  2. DCMHA  - Dynamic Cross-Head Mixed Attention (DCFormer, 2024)
  3. ColMHA - Collaborative Multi-Head Attention (Cordonnier et al., 2021)
  4. MMA    - Mixed Multi-Head Self-Attention (global/local/forward/backward)
  5. MoA    - Mixture of Attention Heads (MoE-style routing)

All expose the same interface as TransformerPC:
    forward(x) -> x
    get_M_matrices() -> []
    get_kl_divergence() -> Tensor(0.)
"""

import math
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

BASELINE_NAMES = ['tha', 'dcmha', 'colmha', 'mma', 'moa']


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

class _FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)


def _causal(T: int, device) -> torch.Tensor:
    m = torch.full((T, T), float('-inf'), device=device)
    return torch.triu(m, diagonal=1)


class _SinPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:x.size(1)].unsqueeze(0)


# ===========================================================================
# 1. THA – Talking-Heads Attention
# ===========================================================================

class THAAttention(nn.Module):
    """Pre/post-softmax H×H head-mixing projections."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.pre_proj  = nn.Linear(n_heads, n_heads, bias=False)
        self.post_proj = nn.Linear(n_heads, n_heads, bias=False)
        self.drop = nn.Dropout(dropout)
        nn.init.eye_(self.pre_proj.weight)
        nn.init.eye_(self.post_proj.weight)

    def forward(self, x, attn_mask=None):
        B, T, d = x.shape
        H, d_h = self.n_heads, self.d_head
        Q = self.W_q(x).view(B, T, H, d_h).transpose(1, 2)
        K = self.W_k(x).view(B, T, H, d_h).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, d_h).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_h)  # [B,H,T,T]
        # Pre mixing: H last -> linear -> H second
        scores = self.pre_proj(scores.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if attn_mask is not None:
            scores = scores + (attn_mask[:, None] if attn_mask.ndim == 3 else attn_mask)
        attn = F.softmax(scores, dim=-1)
        attn = self.post_proj(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.drop(attn)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, d)
        return self.out_proj(out)


# ===========================================================================
# 2. DCMHA – Dynamic Cross-Head Mixed Attention
# ===========================================================================

class DCMHAAttention(nn.Module):
    """Input-dependent H×H head composition via low-rank router + gate."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 router_rank: int = None, **kwargs):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        rk = router_rank or max(4, n_heads // 2)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.router = nn.Sequential(
            nn.Linear(d_model, rk), nn.Tanh(),
            nn.Linear(rk, n_heads * n_heads),
        )
        self.gate_proj = nn.Linear(d_model, n_heads)

    def forward(self, x, attn_mask=None):
        B, T, d = x.shape
        H, d_h = self.n_heads, self.d_head
        Q = self.W_q(x).view(B, T, H, d_h).transpose(1, 2)
        K = self.W_k(x).view(B, T, H, d_h).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, d_h).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_h)  # [B,H,T,T]
        # Dynamic alpha [B,T,H,H], softmax over source-head dim
        alpha = F.softmax(self.router(x).view(B, T, H, H), dim=-2)  # [B,T,H_in,H_out]
        # Compose: new_score[h_out,i,j] = sum_{h_in} alpha[i,h_in,h_out] * score[h_in,i,j]
        scores_ti = scores.permute(0, 2, 1, 3)                           # [B,T,H,T]
        composed = torch.einsum('btih,btis->bths', alpha, scores_ti)     # [B,T,H_out,T]
        composed = composed.permute(0, 2, 1, 3)                          # [B,H,T,T]
        gate = torch.sigmoid(self.gate_proj(x)).permute(0, 2, 1).unsqueeze(-1)  # [B,H,T,1]
        scores_final = gate * composed + (1.0 - gate) * scores
        if attn_mask is not None:
            scores_final = scores_final + (attn_mask[:, None] if attn_mask.ndim == 3 else attn_mask)
        attn = self.drop(F.softmax(scores_final, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, d)
        return self.out_proj(out)


# ===========================================================================
# 3. ColMHA – Collaborative Multi-Head Attention
# ===========================================================================

class CollaborativeAttention(nn.Module):
    """Shared Q/K projection bases with per-head H×H combination coefficients."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.W_shared_q = nn.Linear(d_model, d_model, bias=False)
        self.W_shared_k = nn.Linear(d_model, d_model, bias=False)
        self.combine_q = nn.Parameter(torch.eye(n_heads) + 0.01 * torch.randn(n_heads, n_heads))
        self.combine_k = nn.Parameter(torch.eye(n_heads) + 0.01 * torch.randn(n_heads, n_heads))
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, T, d = x.shape
        H, d_h = self.n_heads, self.d_head
        Q_s = self.W_shared_q(x).view(B, T, H, d_h)
        K_s = self.W_shared_k(x).view(B, T, H, d_h)
        # combine_q[h_out, h_in]: each output head is a combination of shared projections
        Q = torch.einsum('oi,btid->btod', self.combine_q, Q_s).transpose(1, 2)  # [B,H,T,d_h]
        K = torch.einsum('oi,btid->btod', self.combine_k, K_s).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, d_h).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_h)
        if attn_mask is not None:
            scores = scores + (attn_mask[:, None] if attn_mask.ndim == 3 else attn_mask)
        attn = self.drop(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, d)
        return self.out_proj(out)


# ===========================================================================
# 4. MMA – Mixed Multi-Head Self-Attention
# ===========================================================================

class MMAAttention(nn.Module):
    """Heads divided into 4 structural role groups: global/local/forward/backward."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 local_window: int = 8, **kwargs):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.local_window = local_window
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        groups = ['global', 'local', 'forward', 'backward']
        self.head_groups = [groups[h % 4] for h in range(n_heads)]

    def _mask(self, group: str, T: int, device) -> torch.Tensor:
        if group in ('global', 'forward'):
            return _causal(T, device)
        elif group == 'local':
            m = torch.full((T, T), float('-inf'), device=device)
            for i in range(T):
                m[i, max(0, i - self.local_window + 1):i + 1] = 0.0
            return m
        else:  # backward
            m = torch.full((T, T), float('-inf'), device=device)
            return torch.tril(m, diagonal=-1)

    def forward(self, x, attn_mask=None):
        B, T, d = x.shape
        H, d_h = self.n_heads, self.d_head
        Q = self.W_q(x).view(B, T, H, d_h).transpose(1, 2)
        K = self.W_k(x).view(B, T, H, d_h).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, d_h).transpose(1, 2)
        base = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_h)
        outs = []
        for h in range(H):
            s = base[:, h] + self._mask(self.head_groups[h], T, x.device).unsqueeze(0)
            if attn_mask is not None:
                s = s + attn_mask
            a = self.drop(F.softmax(s.masked_fill(torch.isneginf(s).all(-1, keepdim=True), 0), dim=-1))
            outs.append(torch.matmul(a, V[:, h]))
        out = torch.stack(outs, dim=2).view(B, T, d)
        return self.out_proj(out)


# ===========================================================================
# 5. MoA – Mixture of Attention Heads
# ===========================================================================

class MoAAttention(nn.Module):
    """n_experts single-head experts; router selects top-k = n_heads per token."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 expert_factor: int = 2, **kwargs):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.n_experts = n_heads * expert_factor
        self.top_k = n_heads
        n_e, d_h, k = self.n_experts, self.d_head, self.top_k
        self.W_q = nn.Parameter(torch.empty(n_e, d_model, d_h))
        self.W_k = nn.Parameter(torch.empty(n_e, d_model, d_h))
        self.W_v = nn.Parameter(torch.empty(n_e, d_model, d_h))
        for w in [self.W_q, self.W_k, self.W_v]:
            nn.init.xavier_uniform_(w.view(n_e * d_model, d_h))
        self.out_proj = nn.Linear(k * d_h, d_model)
        self.router = nn.Linear(d_model, n_e)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, T, d = x.shape
        n_e, d_h, k = self.n_experts, self.d_head, self.top_k
        # Router
        top_logits, top_idx = self.router(x).topk(k, dim=-1)   # [B,T,k]
        gate_w = F.softmax(top_logits, dim=-1)                  # [B,T,k]
        # Causal mask
        causal = _causal(T, x.device)
        if attn_mask is not None:
            causal = causal + attn_mask
        # QKV for all experts
        x_flat = x.reshape(B * T, d)
        Q_all = torch.einsum('bi,eij->bej', x_flat, self.W_q).view(B, T, n_e, d_h)
        K_all = torch.einsum('bi,eij->bej', x_flat, self.W_k).view(B, T, n_e, d_h)
        V_all = torch.einsum('bi,eij->bej', x_flat, self.W_v).view(B, T, n_e, d_h)
        # Batched attention per expert
        Q_e = Q_all.permute(2, 0, 1, 3).reshape(n_e * B, T, d_h)
        K_e = K_all.permute(2, 0, 1, 3).reshape(n_e * B, T, d_h)
        V_e = V_all.permute(2, 0, 1, 3).reshape(n_e * B, T, d_h)
        sc = torch.bmm(Q_e, K_e.transpose(1, 2)) / math.sqrt(d_h) + (causal.repeat(n_e, 1, 1) if causal.ndim == 3 else causal.unsqueeze(0))
        at = self.drop(F.softmax(sc, dim=-1))
        out_e = torch.bmm(at, V_e).view(n_e, B, T, d_h).permute(1, 2, 0, 3)  # [B,T,n_e,d_h]
        # Gather top-k, gate, flatten
        idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, d_h)
        out_topk = torch.gather(out_e, 2, idx_exp)              # [B,T,k,d_h]
        out_gated = (out_topk * gate_w.unsqueeze(-1)).view(B, T, k * d_h)
        return self.out_proj(out_gated)


# ===========================================================================
# Decoder stack shared by all baselines
# ===========================================================================

_ATTN_CLS = {
    'tha':    THAAttention,
    'dcmha':  DCMHAAttention,
    'colmha': CollaborativeAttention,
    'mma':    MMAAttention,
    'moa':    MoAAttention,
}


class BaselineDecoderLayer(nn.Module):
    def __init__(self, baseline: str, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, **attn_kwargs):
        super().__init__()
        self.self_attn = _ATTN_CLS[baseline](d_model, n_heads, dropout, **attn_kwargs)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff  = _FFN(d_model, d_ff, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        mask = _causal(x.size(1), x.device)
        if padding_mask is not None:
            mask = mask.expand(x.size(0), -1, -1).masked_fill(~padding_mask[:, None, :].bool(), float('-inf'))
        x = self.ln1(x + self.drop(self.self_attn(x, attn_mask=mask)))
        x = self.ln2(x + self.drop(self.ff(x)))
        return x


class BaselineDecoder(nn.Module):
    def __init__(self, baseline: str, d_model: int, n_heads: int, d_ff: int,
                 n_layers: int, dropout: float = 0.1, **attn_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            BaselineDecoderLayer(baseline, d_model, n_heads, d_ff, dropout, **attn_kwargs)
            for _ in range(n_layers)
        ])
        self.pe = _SinPE(d_model)

    def forward(self, x, padding_mask=None):
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
        return x

    def get_M_matrices(self): return []


class BaselineTransformer(nn.Module):
    """Drop-in replacement for TransformerPC using a baseline attention module."""

    def __init__(self, baseline: str, d_model: int = 128, n_heads: int = 8,
                 d_ff: int = 256, n_layers: int = 4, dropout: float = 0.1,
                 dev_mode: bool = False, **kwargs):
        super().__init__()
        assert baseline in BASELINE_NAMES, \
            f"Unknown baseline '{baseline}'. Choose from {BASELINE_NAMES}"
        self.baseline  = baseline
        self.d_model   = d_model
        self.dev_mode  = dev_mode
        # Only pass recognised per-baseline kwargs
        _KNOWN = {'router_rank', 'local_window', 'expert_factor'}
        attn_kwargs = {k: v for k, v in kwargs.items() if k in _KNOWN}
        self.decoder = BaselineDecoder(
            baseline, d_model, n_heads, d_ff, n_layers, dropout, **attn_kwargs
        )

    def forward(self, x, padding_mask=None):
        return self.decoder(x, padding_mask=padding_mask)

    def get_M_matrices(self):        return []
    def get_M0_matrices(self):       return []
    def get_off_diag_params(self):   return {}
    def get_kl_divergence(self):
        return torch.tensor(0.0, device=next(self.parameters()).device)
