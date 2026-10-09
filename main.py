"""DAMCHA: Data-Adaptive Mahalanobis Cross-Head Attention.

Usage:
# CV Datasets: cifar10, cifar100, mnist
python main.py --dataset cifar10 --epochs 50
python main.py --dataset cifar10 --epochs 50 --use_M --use_mlp --share_mlp

# Multi-head M mode: sum blocks along each block row
python main.py --dataset cifar10 --epochs 50 --use_M --use_mlp --share_mlp --use_multihead_M

# Off-diagonal modes:
python main.py --dataset cifar10 --epochs 50 --use_M --use_mlp --share_mlp --off_diag_mode linear_comb
python main.py --dataset cifar10 --epochs 50 --use_M --use_mlp --share_mlp --off_diag_mode bayesian --kl_weight 1e-4

# NLP Datasets: wmt, cnn_dailymail, commongen
python main.py --dataset commongen --epochs 50 --use_M --use_mlp --share_mlp
python main.py --dataset wmt --epochs 50 --use_M --use_mlp --share_mlp --use_multihead_M
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader

from data.datasets import get_image_reconstruction_datasets
from train import train
from utils import seed_everything


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='DAMCHA: Data-Adaptive Mahalanobis Cross-Head Attention'
    )
    
    # Mode
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Enable development mode with debug information'
    )
    parser.add_argument(
        '--gpu', type=int, default=0,
        help='CUDA device index'
    )
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data/rawdata',
                       help='Directory for dataset')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'mnist', 'wmt', 'cnn_dailymail', 'commongen'],
                       help='Dataset name (CV: cifar10/cifar100/mnist, NLP: wmt/cnn_dailymail/commongen)')
    parser.add_argument('--img_size', type=int, default=32,
                       help='Image size')
    parser.add_argument('--in_ch', type=int, default=3,
                       help='Number of input channels')
    parser.add_argument('--patch_size', type=int, default=4,
                       help='Patch size for patchification')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Model architecture (decoder-only)
    parser.add_argument('--d_model', type=int, default=128,
                       help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=256,
                       help='Feed-forward dimension')
    parser.add_argument('--n_layers', type=int, default=4,
                       help='Number of decoder layers')
    
    # Attention mode parameters
    parser.add_argument('--use_M', action='store_true',
                       help='Use M-based attention instead of standard Q/K/V')
    parser.add_argument('--use_mlp', action='store_true',
                       help='Enable MLP-based M generation (requires --use_M)')
    parser.add_argument('--share_mlp', action='store_true',
                       help='Share MLP parameters across all decoder blocks (requires --use_mlp)')
    
    # Off-diagonal mode for M matrix generation
    parser.add_argument('--off_diag_mode', type=str, default='mlp',
                       choices=['mlp', 'linear_comb', 'bayesian', 'separate_B'],
                       help='Off-diagonal block generation mode: '
                            'mlp=direct MLP output, '
                            'separate_B=independent learnable B matrix (decoupled from MLP diagonal), '
                            'linear_comb=linear combination of diagonal blocks, '
                            'bayesian=linear_comb with distribution inference')
    parser.add_argument('--kl_weight', type=float, default=1e-4,
                       help='Weight for KL divergence term in Bayesian mode (ELBO loss)')
    parser.add_argument('--diag_scale', type=float, default=1.0,
                       help='Scaling factor for diagonal blocks (default 1.0)')
    parser.add_argument('--off_diag_scale', type=float, default=1.0,
                       help='Scaling factor for off-diagonal blocks (controls off-diagonal strength)')
    parser.add_argument('--off_diag_alpha_init', type=float, default=0.1,
                       help='Initial std for off-diagonal alpha weights (linear combination coefficients)')
    parser.add_argument('--use_layer_bias', action='store_true', default=False,
                       help='Use learnable per-layer bias for M matrix (enables different M per layer with shared MLP)')
    parser.add_argument('--no_layer_bias', action='store_true',
                       help='Disable per-layer bias for M matrix')
    parser.add_argument('--layer_bias_rank', type=int, default=16,
                       help='Rank for low-rank layer bias factorization (higher = more expressive)')
    
    # Multi-head M mode: aggregate off-diagonal to diagonal, then extract per-head M
    parser.add_argument('--use_multihead_M', action='store_true',
                       help='Compatibility flag; DAMCHA always uses per-head RowSum metrics')
    
    # MLP hidden dimensions
    parser.add_argument('--mlp_hidden_1', type=int, default=256,
                       help='First hidden layer size for MLP')
    parser.add_argument('--mlp_hidden_2', type=int, default=512,
                       help='Second hidden layer size for MLP')
    parser.add_argument('--mlp_hidden_dims', type=str, default=None,
                       help='Comma-separated hidden layer dimensions (e.g., "256,512"). Overrides mlp_hidden_1 and mlp_hidden_2 if provided.')
    
    # Self-supervised learning
    parser.add_argument('--mask_ratio', type=float, default=0.15,
                       help='Ratio of tokens to mask')

    # Top-K metrics for detailed analysis
    parser.add_argument('--top_k_samples', type=int, default=50,
                       help='Number of top samples to select for detailed metrics')
    parser.add_argument('--no_topk_metrics', action='store_true',
                       help='Disable detailed Top-K metrics computation (Top-MSE FID & Top-FID MSE)')
    
    # Block ablation evaluation
    parser.add_argument('--run_block_ablation', action='store_true',
                       help='Run diagonal-only and off-diagonal-only evaluation after training')

    # Free-form M ablation: skip block masking, unconstrained D×D M
    parser.add_argument('--free_M', action='store_true',
                       help='Ablation: skip block-diagonal structure masking, '
                            'let MLP output a fully unconstrained D\u00d7D M matrix. '
                            'Same params as default but no structural inductive bias.')

    # Compact M-attention: low-rank factored MLP-M for fair param-count comparison
    parser.add_argument('--compact_M_rank', type=int, default=0,
                       help='If >0, MLP-M outputs 2*D*r values forming M=U@V^T (rank-r). '
                            'Reduces params to match baselines. Default 0 = full D×D M.')

    # Baseline model selection
    parser.add_argument('--baseline', type=str, default=None,
                       choices=['tha', 'dcmha', 'colmha', 'mma', 'moa'],
                       help='Use a baseline attention instead of DAMCHA: '
                            'tha=Talking-Heads, dcmha=Dynamic Cross-Head (DCFormer), '
                            'colmha=Collaborative MHA, mma=Mixed Multi-Head, '
                            'moa=Mixture of Attention Heads')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Output directory (structure: outputs/{dataset}/visualizations/{mode}/, outputs/{dataset}/metrics/{mode}/)
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Base directory for outputs (visualizations and metrics)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    
    args = parser.parse_args()
    if args.d_model <= 0 or args.n_heads <= 0 or args.d_model % args.n_heads:
        parser.error('d_model must be positive and divisible by n_heads')
    if args.n_layers < 1 or args.epochs < 1 or args.batch_size < 1 or args.save_every < 1:
        parser.error('layers, epochs, batch_size and save_every must be positive')
    if args.dataset == 'mnist':
        args.in_ch = 1
    if args.use_mlp and not args.use_M:
        parser.error('--use_mlp requires --use_M')
    if args.share_mlp and not args.use_mlp:
        parser.error('--share_mlp requires --use_mlp')
    if args.baseline and args.use_M:
        parser.error('--baseline and --use_M select different attention mechanisms')
    return args


def print_config(config: dict):
    """Print configuration in a formatted way."""
    print("\n" + "="*60)
    print("DAMCHA Configuration")
    print("="*60)
    
    print("\n[Mode]")
    print(f"  Development mode: {config['dev_mode']}")
    print(f"  Device: {config['device']}")
    print(f"  Random seed: {config['seed']}")
    
    print("\n[Data]")
    print(f"  Dataset: {config['dataset']}")
    print(f"  Image size: {config['img_size']}")
    print(f"  Patch size: {config['patch_size']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Mask ratio: {config['mask_ratio']}")
    
    print("\n[Model Architecture (Decoder-Only)]")
    print(f"  d_model: {config['d_model']}")
    print(f"  n_heads: {config['n_heads']}")
    print(f"  d_ff: {config['d_ff']}")
    print(f"  n_layers: {config['n_layers']}")
    
    print("\n[Attention Mode]")
    if config.get('baseline'):
        print(f"  Mode: Baseline – {config['baseline'].upper()}")
    elif not config['use_M']:
        print(f"  Mode: Standard Q/K/V Transformer")
    elif config['use_mlp']:
        mlp_mode = "SHARED" if config['share_mlp'] else "SEPARATE"
        print(f"  Mode: DAMCHA with MLP ({mlp_mode})")
    else:
        print(f"  Mode: Static-metric")
    print(f"  use_M: {config['use_M']}")
    print(f"  use_mlp: {config['use_mlp']}")
    print(f"  share_mlp: {config['share_mlp']}")
    print(f"  use_multihead_M: {config['use_multihead_M']}")
    
    if config['use_M'] and config['use_mlp']:
        print("\n[MLP Parameters]")
        print(f"  MLP hidden layers: {config['mlp_hidden']}")
        print(f"  Share MLP across blocks: {config['share_mlp']}")
        if config['share_mlp']:
            print(f"  Per-layer bias: {config['use_layer_bias']}")
            if config['use_layer_bias']:
                print(f"  Layer bias rank: {config['layer_bias_rank']}")
        print(f"  Off-diagonal mode: {config['off_diag_mode']}")
        print(f"  Diagonal scale: {config['diag_scale']}")
        print(f"  Off-diagonal scale: {config['off_diag_scale']}")
        if config['off_diag_mode'] in ['linear_comb', 'bayesian']:
            print(f"  Off-diagonal alpha init: {config['off_diag_alpha_init']}")
        if config['off_diag_mode'] == 'bayesian':
            print(f"  KL weight (ELBO): {config['kl_weight']}")
    
    print("\n[Training]")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Weight decay: {config['weight_decay']}")
    print(f"  Save directory: {config['save_dir']}")
    print(f"  Save every: {config['save_every']} epochs")
    
    print("\n" + "="*60 + "\n")


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    seed_everything(args.seed)
    
    # Check device and set GPU
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = 'cpu'
    elif args.device == 'cuda':
        args.device = f'cuda:{args.gpu}'
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    
    # Create save directories (output subdirs created in train.py based on dataset)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build configuration dictionary
    config = {
        # Mode
        'dev_mode': args.dev,
        'device': args.device,
        'seed': args.seed,
        
        # Data
        'dataset': args.dataset,
        'data_dir': args.data_dir,
        'img_size': args.img_size,
        'in_ch': args.in_ch,
        'patch_size': args.patch_size,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'mask_ratio': args.mask_ratio,
        
        # Model (decoder-only)
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'd_ff': args.d_ff,
        'n_layers': args.n_layers,
        
        # Attention mode
        'use_M': args.use_M,
        'use_mlp': args.use_mlp if args.use_M else False,
        'share_mlp': args.share_mlp if (args.use_M and args.use_mlp) else False,
        
        # MLP parameters
        'mlp_hidden': () if (args.mlp_hidden_dims is not None and args.mlp_hidden_dims.strip() == '') else (tuple(map(int, args.mlp_hidden_dims.split(','))) if args.mlp_hidden_dims else (args.mlp_hidden_1, args.mlp_hidden_2)),
        
        # Off-diagonal mode
        'off_diag_mode': args.off_diag_mode if (args.use_M and args.use_mlp) else 'mlp',
        'kl_weight': args.kl_weight,
        'diag_scale': args.diag_scale,
        'off_diag_scale': args.off_diag_scale,
        'off_diag_alpha_init': args.off_diag_alpha_init,
        'use_layer_bias': (args.use_layer_bias and not args.no_layer_bias) if (args.use_M and args.use_mlp and args.share_mlp) else False,
        'layer_bias_rank': args.layer_bias_rank,
        
        # Multi-head M mode
        'use_multihead_M': args.use_M,

        # Compact M rank (0 = full D×D, >0 = low-rank U@V^T)
        'compact_rank': args.compact_M_rank if (args.use_M and args.use_mlp) else 0,

        # Free-form M ablation (no block structure constraint)
        'free_M': args.free_M if (args.use_M and args.use_mlp) else False,
        
        # Training
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'save_dir': args.save_dir,
        'save_every': args.save_every,
        'resume': args.resume,
        
        # Output directory
        'output_dir': args.output_dir,

        # Top-K metrics
        'top_k_samples': args.top_k_samples,
        'compute_topk_metrics': not args.no_topk_metrics,
        
        # Block ablation evaluation
        'run_block_ablation': args.run_block_ablation,

        # Baseline
        'baseline': args.baseline,
    }
    
    # Print configuration
    print_config(config)
    
    # Determine if CV or NLP dataset
    cv_datasets = ['cifar10', 'cifar100', 'mnist']
    nlp_datasets = ['wmt', 'cnn_dailymail', 'commongen']
    
    dataset_name = config['dataset'].lower()
    
    if dataset_name in cv_datasets:
        # CV dataset: image reconstruction
        print(f"Loading CV dataset: {config['dataset']}...")
        train_loader, val_loader = get_image_reconstruction_datasets(
            dataset_name=config['dataset'],
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            img_size=config['img_size'],
            num_workers=config['num_workers'],
            train_split=0.7
        )
        
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Batches per epoch: {len(train_loader)}\n")
        
        # Start CV training
        print("Starting CV training...\n")
        train(train_loader, val_loader, config)
        
    elif dataset_name in nlp_datasets:
        # NLP dataset: sequence-to-sequence
        from data.datasets_nlp import get_nlp_dataloader
        from train_nlp import train_nlp
        
        print(f"Loading NLP dataset: {config['dataset']}...")
        train_loader, val_loader, tokenizer = get_nlp_dataloader(
            dataset_name=config['dataset'],
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
        )
        
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Vocab size: {tokenizer.vocab_size}")
        print(f"Batches per epoch: {len(train_loader)}\n")
        
        # Start NLP training
        print("Starting NLP training...\n")
        train_nlp(train_loader, val_loader, tokenizer, config)
    
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")
    
    print("\nTraining finished!")


if __name__ == '__main__':
    main()
