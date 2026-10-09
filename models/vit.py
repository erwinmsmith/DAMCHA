"""
Vision Transformer (ViT) with optional M-attention support.

This module implements a Vision Transformer that can use either:
- Standard self-attention (Q/K/V)
- M-based attention with MLP generation
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .m_attention import MBasedAttention, create_shared_mlp


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings."""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, n_patches_h, n_patches_w)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class ViTBlock(nn.Module):
    """Transformer block for ViT with optional M-attention."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        use_M: bool = False,
        use_mlp: bool = False,
        shared_mlp: Optional[nn.Module] = None,
        off_diag_mode: str = 'mlp',
        mlp_hidden: Tuple[int, ...] = (256, 512),
    ):
        super().__init__()
        self.use_M = use_M
        
        if use_M:
            # Use M-attention
            self.attn = MBasedAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                use_M=True,
                use_mlp=use_mlp,
                mlp_hidden_dims=mlp_hidden,
                off_diag_mode=off_diag_mode,
                shared_mlp=shared_mlp,
            )
            # Set shared MLP if provided
            if use_mlp and shared_mlp is not None:
                self.attn.set_shared_mlp(shared_mlp)
        else:
            # Standard attention
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # MLP
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x, head_metrics=None):
        # x: (B, N, d_model)
        if self.use_M:
            # M-attention returns (output, attention_weights)
            attn_out, _ = self.attn(self.norm1(x), head_metrics=head_metrics)
            x = x + attn_out
        else:
            # Standard attention
            attn_out, _ = self.attn(
                self.norm1(x),
                self.norm1(x),
                self.norm1(x),
            )
            x = x + attn_out
        
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for image classification."""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        n_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        use_M: bool = False,
        use_mlp: bool = False,
        share_mlp: bool = False,
        mlp_hidden: Tuple[int, ...] = (256, 512),
        off_diag_mode: str = 'mlp',
    ):
        super().__init__()
        self.use_M = use_M
        self.use_mlp = use_mlp
        self.share_mlp = share_mlp
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        n_patches = self.patch_embed.n_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Create shared MLP if needed
        shared_mlp = None
        if use_M and use_mlp and share_mlp:
            shared_mlp = create_shared_mlp(
                d_model=embed_dim,
                n_heads=n_heads,
                hidden_dims=mlp_hidden,
                dropout=dropout,
                off_diag_mode=off_diag_mode,
            )
        
        # Transformer blocks
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            block = ViTBlock(
                d_model=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_M=use_M,
                use_mlp=use_mlp,
                shared_mlp=shared_mlp,
                off_diag_mode=off_diag_mode,
                mlp_hidden=mlp_hidden,
            )
            # Update MLP hidden dims if using M-attention
            if use_M and use_mlp:
                if shared_mlp is not None:
                    block.attn.set_shared_mlp(shared_mlp)
            self.blocks.append(block)
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # x: (B, C, H, W)
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches+1, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        metrics = None
        if self.use_M and self.use_mlp and self.share_mlp:
            metrics = self.blocks[0].attn.mlp_m.head_metrics(x)
        for block in self.blocks:
            x = block(x, head_metrics=metrics)
        
        # Classification head
        x = self.norm(x)
        cls_token_final = x[:, 0]  # (B, embed_dim)
        logits = self.head(cls_token_final)  # (B, num_classes)
        
        return logits


def create_vit_tiny(num_classes: int = 1000, **kwargs):
    """ViT-Tiny: 5.7M params."""
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        n_heads=3,
        mlp_ratio=4,
        num_classes=num_classes,
        **kwargs
    )


def create_vit_small(num_classes: int = 1000, **kwargs):
    """ViT-Small: 22M params."""
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        n_heads=6,
        mlp_ratio=4,
        num_classes=num_classes,
        **kwargs
    )


def create_vit_base(num_classes: int = 1000, **kwargs):
    """ViT-Base: 86M params."""
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4,
        num_classes=num_classes,
        **kwargs
    )


def create_vit_large(num_classes: int = 1000, **kwargs):
    """ViT-Large: 307M params."""
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        n_heads=16,
        mlp_ratio=4,
        num_classes=num_classes,
        **kwargs
    )
