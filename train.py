"""
Training script for DAMCHA Transformer with self-supervised learning.

Single-phase training:
- Decoder-only architecture with DAMCHA attention
- Gradient flows from loss to all parameters
- MLP-based M generation with structural constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from typing import Tuple, Optional

from models.transformer_damcha import TransformerPC
from models.baselines import BaselineTransformer, BASELINE_NAMES
from utils import (
    AverageMeter, Timer, count_parameters, get_lr,
    save_checkpoint, MaskGenerator, Patchify
)
from visualization import (
    MetricsLogger, Visualizer,
    compute_fid_score, compute_mse, compute_stats,
    compute_topk_metrics, save_topk_samples_data,
    compute_per_sample_metrics, save_all_samples_metrics
)


class ReconstructionHead(nn.Module):
    """
    Reconstruction head for self-supervised learning.
    Maps decoder output back to patch space.
    """
    
    def __init__(self, d_model: int, patch_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, patch_dim),
        )
    
    def forward(self, x):
        return self.proj(x)


class Trainer:
    """Trainer for DAMCHA Transformer with single-phase training."""
    
    def __init__(
        self,
        model: TransformerPC,
        head: ReconstructionHead,
        patcher: Patchify,
        mask_generator: MaskGenerator,
        device: str = 'cuda',
        dev_mode: bool = False,
        visualizer: Visualizer = None,
        top_k_samples: int = 50,
        compute_topk_metrics: bool = False,
        metrics_save_dir: str = './outputs/metrics',
        off_diag_mode: str = 'mlp',
        kl_weight: float = 1e-4,
    ):
        self.model = model
        self.head = head
        self.patcher = patcher
        self.mask_generator = mask_generator
        self.device = device
        self.dev_mode = dev_mode
        self.visualizer = visualizer
        self.top_k_samples = top_k_samples
        self.compute_topk_metrics = compute_topk_metrics
        self.metrics_save_dir = metrics_save_dir
        self.off_diag_mode = off_diag_mode
        self.kl_weight = kl_weight

        # Move to device
        self.model.to(device)
        self.head.to(device)
        self.patcher.to(device)
    
    def _compute_reconstruction_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask_bool: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss only on masked positions.
        
        Args:
            pred: Predicted patches [B, T, patch_dim]
            target: Target patches [B, T, patch_dim]
            mask_bool: Boolean mask [B, T]
            
        Returns:
            Scalar loss
        """
        # Only compute loss on masked positions
        pred_masked = pred[mask_bool]
        target_masked = target[mask_bool]
        
        loss = F.mse_loss(pred_masked, target_masked)
        
        return loss
    
    def train_epoch(
        self,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        epoch: int,
    ) -> Tuple[float, float]:
        """
        Train one epoch (single-phase training).
        
        Args:
            loader: Data loader
            optimizer: Optimizer
            epoch: Current epoch
            
        Returns:
            Tuple of (average_loss, epoch_time)
        """
        self.model.train()
        self.head.train()
        
        loss_meter = AverageMeter()
        timer = Timer()
        timer.start()
        
        for batch_idx, (imgs, _) in enumerate(loader):
            imgs = imgs.to(self.device)
            B = imgs.size(0)
            
            # Convert to patches
            patches = self.patcher.to_sequence(imgs)  # [B, T, d_model]
            T = patches.size(1)
            
            # Generate row-wise random mask
            mask_indices, mask_bool = self.mask_generator.generate_mask(
                B, T, self.device
            )
            
            # Apply mask to patches
            patches_masked = self.mask_generator.apply_mask(patches, mask_bool)
            
            # Get target patches (raw pixel values)
            with torch.no_grad():
                imgs_flat = imgs.view(B, self.patcher.in_ch, -1)
                imgs_flat = imgs_flat.view(
                    B, self.patcher.in_ch,
                    self.patcher.grid, self.patcher.patch,
                    self.patcher.grid, self.patcher.patch
                )
                imgs_flat = imgs_flat.permute(0, 2, 4, 1, 3, 5).contiguous()
                target_patches = imgs_flat.view(
                    B, T, self.patcher.in_ch * self.patcher.patch ** 2
                )
            
            # Forward pass
            optimizer.zero_grad()
            
            # Decoder-only forward
            dec_out = self.model(patches_masked)
            
            # Reconstruction
            pred_patches = self.head(dec_out)  # [B, T, patch_dim]
            
            # Compute loss only on masked positions
            recon_loss = self._compute_reconstruction_loss(
                pred_patches, target_patches, mask_bool
            )
            
            # Add KL divergence for Bayesian mode (ELBO loss)
            if self.off_diag_mode == 'bayesian':
                kl_div = self.model.get_kl_divergence()
                loss = recon_loss + self.kl_weight * kl_div
            else:
                loss = recon_loss
            
            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.head.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            
            # Update metrics
            loss_meter.update(loss.item(), B)
            
            # Print progress
            if batch_idx % 10 == 0 or self.dev_mode:
                lr = get_lr(optimizer)
                print(
                    f"Epoch {epoch} [{batch_idx}/{len(loader)}] "
                    f"Loss: {loss.item():.4f} (avg: {loss_meter.avg:.4f}) "
                    f"LR: {lr:.6f}"
                )
                
                if self.dev_mode:
                    print(f"  Masked rows: {mask_bool.sum().item()} / {B * T}")
                    print(f"  Pred range: [{pred_patches.min():.3f}, {pred_patches.max():.3f}]")
        
        epoch_time = timer.stop()
        return loss_meter.avg, epoch_time
    
    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        epoch: int = 0,
        compute_metrics: bool = True,
    ) -> Tuple[float, float, float, Optional[float], Optional[float], Optional[float], Optional[float], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Evaluate model on validation set.

        Args:
            loader: Validation data loader
            epoch: Current epoch
            compute_metrics: Whether to compute FID and MSE

        Returns:
            Tuple of (val_loss, val_mse, val_fid, top_mse_fid, top_fid_mse, top_mse_best, top_fid_best, original_imgs, masked_imgs, reconstructed_imgs, per_sample_mse, per_sample_fid)
        """
        self.model.eval()
        self.head.eval()
        
        loss_meter = AverageMeter()
        all_originals = []
        all_reconstructed = []
        all_masked = []
        
        # Store first batch for visualization
        first_batch_orig = None
        first_batch_recon = None
        first_batch_masked = None
        
        for batch_idx, (imgs, _) in enumerate(loader):
            imgs = imgs.to(self.device)
            B = imgs.size(0)
            
            # Convert to patches
            patches = self.patcher.to_sequence(imgs)
            T = patches.size(1)
            
            # Generate mask
            mask_indices, mask_bool = self.mask_generator.generate_mask(
                B, T, self.device
            )
            patches_masked = self.mask_generator.apply_mask(patches, mask_bool)
            
            # Target patches
            imgs_flat = imgs.view(B, self.patcher.in_ch, -1)
            imgs_flat = imgs_flat.view(
                B, self.patcher.in_ch,
                self.patcher.grid, self.patcher.patch,
                self.patcher.grid, self.patcher.patch
            )
            imgs_flat = imgs_flat.permute(0, 2, 4, 1, 3, 5).contiguous()
            target_patches = imgs_flat.view(
                B, T, self.patcher.in_ch * self.patcher.patch ** 2
            )
            
            # Forward (decoder-only)
            dec_out = self.model(patches_masked)
            
            pred_patches = self.head(dec_out)
            loss = self._compute_reconstruction_loss(
                pred_patches, target_patches, mask_bool
            )
            
            loss_meter.update(loss.item(), B)
            
            # Reconstruct images for metrics
            if compute_metrics:
                # Simple reconstruction (approximate)
                recon_imgs = self._reconstruct_images_from_patches(pred_patches, imgs.shape)
                masked_imgs = self._create_masked_images(imgs, mask_bool)
                
                all_originals.append(imgs.cpu())
                all_reconstructed.append(recon_imgs.cpu())
                all_masked.append(masked_imgs.cpu())
                
                # Store first batch
                if batch_idx == 0:
                    first_batch_orig = imgs[:8].cpu()
                    first_batch_recon = recon_imgs[:8].cpu()
                    first_batch_masked = masked_imgs[:8].cpu()
            
            # Limit evaluation batches for speed and memory
            if batch_idx >= 50:
                break
        
        val_loss = loss_meter.avg
        val_mse_mean = -1.0; val_mse_q25 = -1.0; val_mse_q75 = -1.0
        val_fid_mean = -1.0; val_fid_q25 = -1.0; val_fid_q75 = -1.0
        top_mse_fid = -1.0
        top_fid_mse = -1.0
        top_mse_best = -1.0
        top_fid_best = -1.0
        per_sample_mse = None
        per_sample_fid = None

        # Compute metrics
        if compute_metrics and len(all_originals) > 0:
            all_originals = torch.cat(all_originals, dim=0)
            all_reconstructed = torch.cat(all_reconstructed, dim=0)

            val_fid_mean = compute_fid_score(all_originals, all_reconstructed, self.device)

            # Compute per-sample metrics for all samples (always computed)
            print("Computing per-sample metrics for all samples...")
            per_sample_mse, per_sample_fid = compute_per_sample_metrics(
                all_originals, all_reconstructed, self.device
            )

            val_mse_mean, val_mse_q25, val_mse_q75 = compute_stats(per_sample_mse)
            _, val_fid_q25, val_fid_q75 = compute_stats(per_sample_fid)

            # Save all samples metrics
            save_all_samples_metrics(per_sample_mse, per_sample_fid, self.metrics_save_dir, epoch)

            # Compute Top-K metrics if enabled
            if self.compute_topk_metrics:
                print(f"Computing Top-{self.top_k_samples} metrics...")
                top_mse_fid, top_fid_mse, top_mse_best, top_fid_best, detailed_info = compute_topk_metrics(
                    all_originals, all_reconstructed, self.top_k_samples, self.device
                )

                # Save detailed Top-K samples data for future analysis
                save_topk_samples_data(detailed_info, self.metrics_save_dir, epoch)

                print(f"  Top-{self.top_k_samples} MSE samples FID: {top_mse_fid:.2f}")
                print(f"  Top-{self.top_k_samples} FID samples MSE: {top_fid_mse:.6f}")
                print(f"  Best single MSE: {top_mse_best:.6f}")
                print(f"  Best single FID: {top_fid_best:.2f}")

        return (val_loss,
                val_mse_mean, val_mse_q25, val_mse_q75,
                val_fid_mean, val_fid_q25, val_fid_q75,
                top_mse_fid, top_fid_mse, top_mse_best, top_fid_best,
                first_batch_orig, first_batch_masked, first_batch_recon,
                per_sample_mse, per_sample_fid)
    
    def _reconstruct_images_from_patches(
        self,
        pred_patches: torch.Tensor,
        target_shape: torch.Size
    ) -> torch.Tensor:
        """
        Reconstruct images from predicted patches (approximate).
        
        Args:
            pred_patches: Predicted patches [B, T, patch_dim]
            target_shape: Target image shape [B, C, H, W]
            
        Returns:
            Reconstructed images [B, C, H, W]
        """
        B, T, patch_dim = pred_patches.shape
        C, H, W = target_shape[1], target_shape[2], target_shape[3]
        p = self.patcher.patch
        grid = self.patcher.grid
        
        # Reshape patches
        patches = pred_patches.view(B, grid, grid, C, p, p)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        imgs = patches.view(B, C, H, W)
        
        return imgs
    
    def _create_masked_images(
        self,
        imgs: torch.Tensor,
        mask_bool: torch.Tensor
    ) -> torch.Tensor:
        """
        Create masked version of images for visualization.
        Masked regions are shown as black (empty).
        
        Args:
            imgs: Original images [B, C, H, W] (normalized to [-1, 1])
            mask_bool: Boolean mask [B, T]
            
        Returns:
            Masked images [B, C, H, W]
        """
        B, C, H, W = imgs.shape
        p = self.patcher.patch
        grid = self.patcher.grid
        
        # Clone to avoid modifying original
        masked_imgs = imgs.clone()
        
        for b in range(B):
            for t in range(mask_bool.size(1)):
                if mask_bool[b, t]:
                    i = t // grid
                    j = t % grid
                    # Set masked patches to black (-1 in normalized space, or 0 in [0,1] space)
                    masked_imgs[b, :, i*p:(i+1)*p, j*p:(j+1)*p] = -1.0
        
        return masked_imgs


def modify_model_scales(model, diag_scale, off_diag_scale):
    """Modify M matrix scales in the model."""
    modified = False
    for module in model.modules():
        if hasattr(module, '_shared_mlp') and module._shared_mlp is not None:
            module._shared_mlp.diag_scale = diag_scale
            module._shared_mlp.off_diag_scale = off_diag_scale
            modified = True
        elif hasattr(module, '_own_mlp') and module._own_mlp is not None:
            module._own_mlp.diag_scale = diag_scale
            module._own_mlp.off_diag_scale = off_diag_scale
            modified = True
    return modified


def run_block_ablation_evaluation(
    model: TransformerPC,
    trainer: 'Trainer',
    val_loader: DataLoader,
    config: dict,
    metrics_dir: str,
    device: str,
):
    """
    Run block ablation evaluation with diagonal-only and off-diagonal-only M matrix.
    
    Args:
        model: Trained TransformerPC model
        trainer: Trainer instance with patcher and head
        val_loader: Validation data loader
        config: Training configuration
        metrics_dir: Directory to save metrics
        device: Device for computation
    """
    import json
    import numpy as np
    
    # Store original scales
    original_diag_scale = config.get('diag_scale', 1.0)
    original_off_diag_scale = config.get('off_diag_scale', 0.5)
    
    results = {}
    
    # Evaluate with diagonal-only (diag_scale=1.0, off_diag_scale=0.0)
    print("\n[Diagonal-only] Evaluating with diag_scale=1.0, off_diag_scale=0.0...")
    modify_model_scales(model, diag_scale=1.0, off_diag_scale=0.0)
    
    (diag_loss,
     diag_mse_mean, diag_mse_q25, diag_mse_q75,
     diag_fid_mean, diag_fid_q25, diag_fid_q75,
     _, _, diag_top_mse_best, diag_top_fid_best,
     _, _, _, _, _) = trainer.evaluate(val_loader, epoch=-1, compute_metrics=True)
    
    results['diagonal_only'] = {
        'val_loss': float(diag_loss),
        'val_mse_mean': float(diag_mse_mean), 'val_mse_q25': float(diag_mse_q25), 'val_mse_q75': float(diag_mse_q75),
        'val_fid_mean': float(diag_fid_mean), 'val_fid_q25': float(diag_fid_q25), 'val_fid_q75': float(diag_fid_q75),
        'top_mse_best': float(diag_top_mse_best), 'top_fid_best': float(diag_top_fid_best),
    }
    print(f"  val_fid_mean: {diag_fid_mean:.2f} [Q25={diag_fid_q25:.2f}, Q75={diag_fid_q75:.2f}]")
    
    # Evaluate with off-diagonal-only (diag_scale=0.0, off_diag_scale=1.0)
    print("\n[Off-diagonal-only] Evaluating with diag_scale=0.0, off_diag_scale=1.0...")
    modify_model_scales(model, diag_scale=0.0, off_diag_scale=1.0)
    
    (offdiag_loss,
     offdiag_mse_mean, offdiag_mse_q25, offdiag_mse_q75,
     offdiag_fid_mean, offdiag_fid_q25, offdiag_fid_q75,
     _, _, offdiag_top_mse_best, offdiag_top_fid_best,
     _, _, _, _, _) = trainer.evaluate(val_loader, epoch=-1, compute_metrics=True)
    
    results['offdiag_only'] = {
        'val_loss': float(offdiag_loss),
        'val_mse_mean': float(offdiag_mse_mean), 'val_mse_q25': float(offdiag_mse_q25), 'val_mse_q75': float(offdiag_mse_q75),
        'val_fid_mean': float(offdiag_fid_mean), 'val_fid_q25': float(offdiag_fid_q25), 'val_fid_q75': float(offdiag_fid_q75),
        'top_mse_best': float(offdiag_top_mse_best), 'top_fid_best': float(offdiag_top_fid_best),
    }
    print(f"  val_fid_mean: {offdiag_fid_mean:.2f} [Q25={offdiag_fid_q25:.2f}, Q75={offdiag_fid_q75:.2f}]")
    
    # Restore original scales
    modify_model_scales(model, diag_scale=original_diag_scale, off_diag_scale=original_off_diag_scale)
    
    # Save results
    results_file = os.path.join(metrics_dir, 'block_ablation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBlock ablation results saved to: {results_file}")
    print("="*80)


def build_model(
    d_model: int,
    n_heads: int,
    d_ff: int,
    n_layers: int,
    use_M: bool,
    use_mlp: bool,
    share_mlp: bool,
    mlp_hidden: Tuple[int, ...],
    off_diag_mode: str,
    diag_scale: float,
    off_diag_scale: float,
    off_diag_alpha_init: float,
    use_layer_bias: bool,
    layer_bias_rank: int,
    dev_mode: bool,
    device: str,
    use_multihead_M: bool = False,
    baseline: str = None,
    compact_rank: int = 0,
    free_M: bool = False,
):
    """
    Build decoder-only model.

    When baseline is set to one of BASELINE_NAMES, a BaselineTransformer is
    returned instead of TransformerPC.  All other M-related parameters are
    ignored for baselines.
    """
    if baseline and baseline in BASELINE_NAMES:
        model = BaselineTransformer(
            baseline=baseline,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=0.1,
            dev_mode=dev_mode,
        )
        return model.to(device)

    model = TransformerPC(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        dropout=0.1,
        dev_mode=dev_mode,
        use_M=use_M,
        use_mlp=use_mlp,
        share_mlp=share_mlp,
        mlp_hidden=mlp_hidden,
        off_diag_mode=off_diag_mode,
        diag_scale=diag_scale,
        off_diag_scale=off_diag_scale,
        off_diag_alpha_init=off_diag_alpha_init,
        use_layer_bias=use_layer_bias,
        layer_bias_rank=layer_bias_rank,
        use_multihead_M=use_multihead_M,
        compact_rank=compact_rank,
        free_M=free_M,
    )

    return model.to(device)


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
):
    """
    Main training function.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
    """
    device = config['device']
    dev_mode = config['dev_mode']
    
    # Get dataset name for directory structure
    dataset_name = config.get('dataset', 'cifar10')
    
    # Determine mode name for output directories
    baseline = config.get('baseline')
    if baseline and baseline in BASELINE_NAMES:
        mode_name = f'baselines/{baseline}'
    elif not config['use_M']:
        mode_name = 'standard'  # Standard Q/K/V Transformer
    elif config['use_mlp']:
        off_diag_mode = config.get('off_diag_mode', 'mlp')
        # Add MLP depth info to mode name
        mlp_hidden = config.get('mlp_hidden', ())
        mlp_depth = len(mlp_hidden) + 1  # +1 for output layer
        depth_suffix = f'_depth{mlp_depth}'
        
        # Add multihead_M suffix if enabled
        multihead_suffix = '_multihead' if config.get('use_multihead_M', False) else ''
        # Add compact rank suffix if enabled
        cr = config.get('compact_rank', 0)
        compact_suffix = f'_compact{cr}' if cr > 0 else ''
        # Add free_M suffix if enabled
        free_suffix = '_freeM' if config.get('free_M', False) else ''

        if config['share_mlp']:
            if off_diag_mode == 'mlp':
                mode_name = f'mlp_shared{depth_suffix}{multihead_suffix}{compact_suffix}{free_suffix}'
            else:
                mode_name = f'mlp_shared_{off_diag_mode}{depth_suffix}{multihead_suffix}{compact_suffix}{free_suffix}'
        else:
            if off_diag_mode == 'mlp':
                mode_name = f'mlp_separate{depth_suffix}{multihead_suffix}{compact_suffix}{free_suffix}'
            else:
                mode_name = f'mlp_separate_{off_diag_mode}{depth_suffix}{multihead_suffix}{compact_suffix}{free_suffix}'
    else:
        mode_name = 'M0_only'   # M0-only attention
    
    # Initialize visualization and metrics logging with dataset and mode-specific paths
    # Directory structure: outputs/{dataset}/visualizations/{mode}/
    #                      outputs/{dataset}/metrics/{mode}/
    #                      checkpoints/{dataset}/{mode}/
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_mode = mode_name.replace('/', '_')
    experiment_name = f"{safe_mode}_{timestamp}"
    
    # Create dataset and mode-specific output directories
    # outputs/{dataset}/visualizations/{mode}/ and outputs/{dataset}/metrics/{mode}/
    output_base = config.get('output_dir', './outputs')
    vis_dir = os.path.join(output_base, dataset_name, 'visualizations', mode_name)
    metrics_dir = os.path.join(output_base, dataset_name, 'metrics', mode_name)
    save_dir = os.path.join(config.get('save_dir', './checkpoints'), dataset_name, mode_name)
    os.makedirs(save_dir, exist_ok=True)
    
    visualizer = Visualizer(
        save_dir=vis_dir,
        experiment_name=experiment_name
    )
    
    metrics_logger = MetricsLogger(
        save_dir=metrics_dir,
        experiment_name=experiment_name
    )
    
    print(f"\n[Output Directories]")
    print(f"  Dataset: {dataset_name}")
    print(f"  Mode: {mode_name}")
    print(f"  Checkpoints: {save_dir}")
    print(f"  Visualizations: {vis_dir}")
    print(f"  Metrics: {metrics_dir}\n")
    
    # Build model
    model = build_model(
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        n_layers=config['n_layers'],
        use_M=config['use_M'],
        use_mlp=config['use_mlp'],
        share_mlp=config['share_mlp'],
        mlp_hidden=config['mlp_hidden'],
        baseline=config.get('baseline'),
        off_diag_mode=config.get('off_diag_mode', 'mlp'),
        diag_scale=config.get('diag_scale', 1.0),
        off_diag_scale=config.get('off_diag_scale', 0.5),
        off_diag_alpha_init=config.get('off_diag_alpha_init', 0.1),
        use_layer_bias=config.get('use_layer_bias', True),
        layer_bias_rank=config.get('layer_bias_rank', 16),
        dev_mode=dev_mode,
        device=device,
        use_multihead_M=config.get('use_multihead_M', False),
        compact_rank=config.get('compact_rank', 0),
        free_M=config.get('free_M', False),
    )
    
    # Build reconstruction head
    patch_dim = config['in_ch'] * config['patch_size'] ** 2
    head = ReconstructionHead(config['d_model'], patch_dim).to(device)
    
    # Build patcher
    patcher = Patchify(
        in_ch=config['in_ch'],
        img_size=config['img_size'],
        patch=config['patch_size'],
        d_model=config['d_model']
    )
    
    # Build mask generator (row-wise masking: 1-3 consecutive rows)
    grid_size = config['img_size'] // config['patch_size']
    mask_generator = MaskGenerator(
        mask_ratio=config['mask_ratio'],
        grid_size=grid_size,
        min_rows=1,
        max_rows=3,
        mode='row'
    )
    
    # Print model info
    total_params = count_parameters(model) + count_parameters(head)
    print(f"\nModel initialized:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Device: {device}")
    print(f"  Dev mode: {dev_mode}\n")
    
    # Optimizer
    optimizer = optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['lr'] * 0.01
    )
    
    # Trainer
    trainer = Trainer(
        model, head, patcher, mask_generator,
        device=device,
        dev_mode=dev_mode,
        visualizer=visualizer,
        top_k_samples=config['top_k_samples'],
        compute_topk_metrics=config['compute_topk_metrics'],
        metrics_save_dir=metrics_dir,  # Use experiment-specific metrics directory
        off_diag_mode=config.get('off_diag_mode', 'mlp'),
        kl_weight=config.get('kl_weight', 1e-4),
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"{'='*60}")
        
        # Single-phase training
        train_loss, train_time = trainer.train_epoch(
            train_loader, optimizer, epoch
        )
        
        print(f"\nTraining completed - Loss: {train_loss:.4f}, Time: {train_time:.2f}s")
        
        # Validation with metrics
        (val_loss,
         val_mse_mean, val_mse_q25, val_mse_q75,
         val_fid_mean, val_fid_q25, val_fid_q75,
         top_mse_fid, top_fid_mse, top_mse_best, top_fid_best,
         orig_imgs, masked_imgs, recon_imgs, _, _) = trainer.evaluate(
            val_loader, epoch=epoch, compute_metrics=True
        )
        print(f"Validation - Loss: {val_loss:.4f}, "
              f"MSE: {val_mse_mean:.4f} [Q25={val_mse_q25:.4f}, Q75={val_mse_q75:.4f}], "
              f"FID: {val_fid_mean:.4f} [Q25={val_fid_q25:.4f}, Q75={val_fid_q75:.4f}]")
        if config['compute_topk_metrics']:
            print(f"Top-50 - MSE samples FID: {top_mse_fid:.4f}, FID samples MSE: {top_fid_mse:.6f}")
            print(f"Best Single - MSE: {top_mse_best:.6f}, FID: {top_fid_best:.4f}")
        
        # Get M* matrices for visualization
        M_star_matrices = model.get_M_matrices()
        M_star_list = [M for _, M in M_star_matrices]
        layer_names = [name for name, _ in M_star_matrices]
        
        # Visualize M* matrices
        if len(M_star_list) > 0:
            visualizer.visualize_M_star_matrices(M_star_list, layer_names, epoch)
            visualizer.visualize_M_star_statistics(M_star_list, layer_names, epoch)
        
        # Visualize reconstructions
        if orig_imgs is not None and recon_imgs is not None:
            visualizer.visualize_reconstructions(
                orig_imgs, recon_imgs, masked_imgs, epoch, n_samples=8
            )
        
        # Log metrics to CSV
        metrics_dict = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_time': train_time,
            'val_loss': val_loss,
            'val_mse_mean': val_mse_mean,
            'val_mse_q25':  val_mse_q25,
            'val_mse_q75':  val_mse_q75,
            'val_fid_mean': val_fid_mean,
            'val_fid_q25':  val_fid_q25,
            'val_fid_q75':  val_fid_q75,
            'learning_rate': get_lr(optimizer),
            'top_mse_fid':  top_mse_fid,
            'top_fid_mse':  top_fid_mse,
            'top_mse_best': top_mse_best,
            'top_fid_best': top_fid_best,
        }

        metrics_logger.log_epoch(metrics_dict)
        
        # Plot training curves
        visualizer.plot_training_curves(
            metrics_logger.get_history(),
            save_name=f'training_curves_epoch_{epoch}.png'
        )
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint (use mode-specific save_dir)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(save_dir, 'best_model.pt')
            )
            print(f"Best model saved (val_loss: {val_loss:.4f})")
        
        # Save regular checkpoint
        if epoch % config['save_every'] == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
            )
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
    print(f"Metrics saved to: {metrics_logger.csv_path}")
    print(f"Visualizations saved to: {visualizer.save_dir}")
    
    # Run block ablation evaluation if enabled
    if config.get('run_block_ablation', False) and config['use_M'] and config['use_mlp']:
        print("\n" + "="*80)
        print("Running Block Ablation Evaluation")
        print("="*80)
        
        run_block_ablation_evaluation(
            model=model,
            trainer=trainer,
            val_loader=val_loader,
            config=config,
            metrics_dir=metrics_dir,
            device=device,
        )
