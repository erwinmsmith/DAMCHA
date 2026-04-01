"""
DAMCHA: Dynamic Attention with M-matrix Cross-Head Aggregation

Core models package containing:
- M-based attention mechanism (replaces Q/K with learnable M matrix)
- Structured MLP for M generation with block-diagonal constraints
- Transformer architecture with DAMCHA attention
- Vision Transformer (ViT) with DAMCHA support
"""

from .m_attention import (
    MBasedAttention,
    StructuredMLPForM,
    create_shared_mlp,
    setup_shared_mlp_for_layers,
)
from .transformer_damcha import (
    TransformerPC,
    DAMCHAAttention,
    Decoder,
    DecoderLayer,
)
from .baselines import (
    BaselineTransformer,
    BASELINE_NAMES,
    THAAttention,
    DCMHAAttention,
    CollaborativeAttention,
    MMAAttention,
    MoAAttention,
)
from .vit import (
    VisionTransformer,
    create_vit_tiny,
    create_vit_small,
    create_vit_base,
    create_vit_large,
)

__all__ = [
    # Core M-attention
    'MBasedAttention',
    'StructuredMLPForM',
    'create_shared_mlp',
    'setup_shared_mlp_for_layers',
    # Transformer
    'TransformerPC',
    'DAMCHAAttention',
    'Decoder',
    'DecoderLayer',
    # Baselines
    'BaselineTransformer',
    'BASELINE_NAMES',
    'THAAttention',
    'DCMHAAttention',
    'CollaborativeAttention',
    'MMAAttention',
    'MoAAttention',
    # ViT
    'VisionTransformer',
    'create_vit_tiny',
    'create_vit_small',
    'create_vit_base',
    'create_vit_large',
]
