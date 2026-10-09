"""
Visualization utilities for DAMCHA Transformer training.

Includes:
- M* matrix visualization
- Reconstructed image visualization
- Training metrics plotting
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from typing import List, Dict, Optional, Tuple
from datetime import datetime


class MetricsLogger:
    """
    Logger for training metrics with CSV export.
    Saves metrics incrementally after each epoch.
    """
    
    def __init__(self, save_dir: str, experiment_name: str = None, resume: bool = False):
        """
        Initialize metrics logger.
        
        Args:
            save_dir: Directory to save metrics
            experiment_name: Name of experiment (default: timestamp)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.csv_path = self.save_dir / f"metrics_{experiment_name}.csv"
        self.metrics_history = []
        
        # Initialize CSV with headers (single-phase training)
        self.headers = [
            'epoch', 'train_loss', 'train_time',
            'val_loss',
            'val_mse_mean', 'val_mse_q25', 'val_mse_q75',
            'val_fid_mean', 'val_feature_q25', 'val_feature_q75',
            'learning_rate', 'timestamp',
            'top_mse_fid', 'top_feature_mse', 'top_mse_best', 'top_feature_best'
        ]
        
        if not (resume and self.csv_path.exists()):
            with open(self.csv_path, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=self.headers).writeheader()
        
        print(f"Metrics will be saved to: {self.csv_path}")
    
    def log_epoch(self, metrics: Dict):
        """
        Log metrics for one epoch and save to CSV.
        
        Args:
            metrics: Dictionary containing epoch metrics
        """
        # Add timestamp
        metrics['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Ensure all headers are present
        for header in self.headers:
            if header not in metrics:
                metrics[header] = None
        
        # Append to history
        self.metrics_history.append(metrics)
        
        # Append to CSV file
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(metrics)
    
    def get_history(self) -> List[Dict]:
        """Get full metrics history."""
        return self.metrics_history


class Visualizer:
    """
    Visualizer for DAMCHA Transformer outputs.
    Creates and saves visualizations during training.
    """
    
    def __init__(self, save_dir: str, experiment_name: str = None):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualizations
            experiment_name: Name of experiment
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_name = experiment_name
        self.epoch_dir = None
        
        print(f"Visualizations will be saved to: {self.save_dir}")
    
    def set_epoch(self, epoch: int):
        """Set current epoch for organizing visualizations."""
        self.epoch_dir = self.save_dir / f"epoch_{epoch:04d}"
        self.epoch_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_M_star_matrices(
        self,
        M_star_list: List[torch.Tensor],
        layer_names: List[str],
        epoch: int,
        M0_list: List[torch.Tensor] = None
    ):
        """
        Visualize M* matrices as heatmaps, optionally with M0 initialization.
        
        Args:
            M_star_list: List of M* matrices from different layers
            layer_names: Names of layers
            epoch: Current epoch
            M0_list: Optional list of M0 initialization matrices
        """
        self.set_epoch(epoch)
        
        n_layers = len(M_star_list)
        
        # If M0 is provided, show both M0 and M* in two rows
        if M0_list is not None:
            fig, axes = plt.subplots(2, n_layers, figsize=(5 * n_layers, 10))
            
            # Plot M0 (top row)
            for idx, (M0, name) in enumerate(zip(M0_list, layer_names)):
                M0_np = M0.detach().cpu().numpy()
                ax = axes[0, idx] if n_layers > 1 else axes[0]
                vabs = max(abs(M0_np.min()), abs(M0_np.max())) + 1e-8
                im = ax.imshow(M0_np, cmap='RdBu_r', aspect='auto', vmin=-vabs, vmax=vabs)
                ax.set_title(f'M0 - {name}\nShape: {M0_np.shape}')
                ax.set_xlabel('Column Index')
                ax.set_ylabel('Row Index')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Plot M* (bottom row)
            for idx, (M_star, name) in enumerate(zip(M_star_list, layer_names)):
                M_np = M_star.detach().cpu().numpy()
                ax = axes[1, idx] if n_layers > 1 else axes[1]
                vabs = max(abs(M_np.min()), abs(M_np.max())) + 1e-8
                im = ax.imshow(M_np, cmap='RdBu_r', aspect='auto', vmin=-vabs, vmax=vabs)
                ax.set_title(f'M* - {name}\nShape: {M_np.shape}')
                ax.set_xlabel('Column Index')
                ax.set_ylabel('Row Index')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            plt.suptitle(f'M0 vs M* Matrices - Epoch {epoch}', fontsize=16, y=0.995)
        else:
            # Original behavior: only show M*
            fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 5))
            
            if n_layers == 1:
                axes = [axes]
            
            for idx, (M_star, name) in enumerate(zip(M_star_list, layer_names)):
                M_np = M_star.detach().cpu().numpy()
                ax = axes[idx]
                vabs = max(abs(M_np.min()), abs(M_np.max())) + 1e-8
                im = ax.imshow(M_np, cmap='RdBu_r', aspect='auto', vmin=-vabs, vmax=vabs)
                ax.set_title(f'{name}\nShape: {M_np.shape}')
                ax.set_xlabel('Column Index')
                ax.set_ylabel('Row Index')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            plt.suptitle(f'M* Matrices - Epoch {epoch}', fontsize=16, y=1.02)
        
        plt.tight_layout()
        
        save_path = self.epoch_dir / f'M_star_matrices_epoch_{epoch:04d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved M* matrices visualization: {save_path}")
    
    def visualize_M_star_statistics(
        self,
        M_star_list: List[torch.Tensor],
        layer_names: List[str],
        epoch: int
    ):
        """
        Visualize statistics of M* matrices (eigenvalues, norms, etc.).
        
        Args:
            M_star_list: List of M* matrices
            layer_names: Names of layers
            epoch: Current epoch
        """
        self.set_epoch(epoch)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Collect statistics
        norms = []
        max_vals = []
        min_vals = []
        eigenvalues_list = []
        
        for M_star in M_star_list:
            M_np = M_star.detach().cpu().numpy()
            norms.append(np.linalg.norm(M_np, 'fro'))
            max_vals.append(M_np.max())
            min_vals.append(M_np.min())
            
            # Compute eigenvalues
            try:
                eigvals = np.linalg.eigvals(M_np)
                eigenvalues_list.append(eigvals)
            except:
                eigenvalues_list.append(None)
        
        # Plot 1: Frobenius norms
        axes[0, 0].bar(range(len(norms)), norms, color='steelblue')
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_ylabel('Frobenius Norm')
        axes[0, 0].set_title('M* Matrix Norms')
        axes[0, 0].set_xticks(range(len(layer_names)))
        axes[0, 0].set_xticklabels(layer_names, rotation=45)
        
        # Plot 2: Max/Min values
        x = range(len(max_vals))
        axes[0, 1].plot(x, max_vals, 'o-', label='Max', color='red')
        axes[0, 1].plot(x, min_vals, 'o-', label='Min', color='blue')
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0, 1].set_xlabel('Layer Index')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].set_title('M* Value Range')
        axes[0, 1].legend()
        axes[0, 1].set_xticks(range(len(layer_names)))
        axes[0, 1].set_xticklabels(layer_names, rotation=45)
        
        # Plot 3: Eigenvalue distribution (first layer)
        if eigenvalues_list[0] is not None:
            eigvals = eigenvalues_list[0]
            axes[1, 0].scatter(eigvals.real, eigvals.imag, alpha=0.6, s=20)
            axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
            axes[1, 0].set_xlabel('Real Part')
            axes[1, 0].set_ylabel('Imaginary Part')
            axes[1, 0].set_title(f'Eigenvalues - {layer_names[0]}')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Eigenvalue magnitudes
        axes[1, 1].set_title('Eigenvalue Magnitudes')
        for idx, (eigvals, name) in enumerate(zip(eigenvalues_list, layer_names)):
            if eigvals is not None:
                mags = np.abs(eigvals)
                mags_sorted = np.sort(mags)[::-1]
                axes[1, 1].plot(mags_sorted, label=name, alpha=0.7)
        axes[1, 1].set_xlabel('Index (sorted)')
        axes[1, 1].set_ylabel('Magnitude')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'M* Statistics - Epoch {epoch}', fontsize=16)
        plt.tight_layout()
        
        save_path = self.epoch_dir / f'M_star_statistics_epoch_{epoch:04d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved M* statistics: {save_path}")
    
    def visualize_reconstructions(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        masked: torch.Tensor,
        epoch: int,
        n_samples: int = 8
    ):
        """
        Visualize original, masked, and reconstructed images.
        
        Args:
            original: Original images [B, C, H, W]
            reconstructed: Reconstructed images [B, C, H, W]
            masked: Masked images [B, C, H, W]
            epoch: Current epoch
            n_samples: Number of samples to visualize
        """
        self.set_epoch(epoch)
        
        n_samples = min(n_samples, original.size(0))
        
        fig, axes = plt.subplots(3, n_samples, figsize=(2 * n_samples, 6))
        
        for i in range(n_samples):
            # Original
            img_orig = original[i].detach().cpu().permute(1, 2, 0).numpy()
            img_orig = (img_orig * 0.5 + 0.5).clip(0, 1)  # Denormalize
            axes[0, i].imshow(img_orig)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)
            
            # Masked
            img_masked = masked[i].detach().cpu().permute(1, 2, 0).numpy()
            img_masked = (img_masked * 0.5 + 0.5).clip(0, 1)
            axes[1, i].imshow(img_masked)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Masked', fontsize=10)
            
            # Reconstructed
            img_recon = reconstructed[i].detach().cpu().permute(1, 2, 0).numpy()
            img_recon = (img_recon * 0.5 + 0.5).clip(0, 1)
            axes[2, i].imshow(img_recon)
            axes[2, i].axis('off')
            if i == 0:
                axes[2, i].set_title('Reconstructed', fontsize=10)
        
        plt.suptitle(f'Image Reconstruction - Epoch {epoch}', fontsize=14)
        plt.tight_layout()
        
        save_path = self.epoch_dir / f'reconstructions_epoch_{epoch:04d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved reconstructions: {save_path}")
    
    def plot_training_curves(
        self,
        metrics_history: List[Dict],
        save_name: str = 'training_curves.png'
    ):
        """
        Plot training curves from metrics history.
        
        Args:
            metrics_history: List of metric dictionaries
            save_name: Name of saved plot
        """
        if not metrics_history:
            return
        
        epochs = [m['epoch'] for m in metrics_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Training loss
        train_loss = [m.get('train_loss', None) for m in metrics_history]
        val_loss = [m.get('val_loss', None) for m in metrics_history]
        
        axes[0, 0].plot(epochs, train_loss, 'o-', label='Train Loss', color='blue')
        axes[0, 0].plot(epochs, val_loss, 'o-', label='Val Loss', color='orange')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: MSE (mean ± IQR)
        val_mse_mean = [m.get('val_mse_mean', None) for m in metrics_history]
        val_mse_q25  = [m.get('val_mse_q25',  None) for m in metrics_history]
        val_mse_q75  = [m.get('val_mse_q75',  None) for m in metrics_history]
        if any(v is not None for v in val_mse_mean):
            axes[0, 1].plot(epochs, val_mse_mean, 'o-', color='green', label='mean')
            if any(v is not None for v in val_mse_q25):
                axes[0, 1].fill_between(epochs, val_mse_q25, val_mse_q75,
                                        alpha=0.25, color='green', label='Q25-Q75')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MSE')
            axes[0, 1].set_title('Validation MSE (mean ± IQR)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: FID (mean ± IQR)
        val_fid_mean = [m.get('val_fid_mean', None) for m in metrics_history]
        val_feature_q25  = [m.get('val_feature_q25',  None) for m in metrics_history]
        val_feature_q75  = [m.get('val_feature_q75',  None) for m in metrics_history]
        if any(v is not None for v in val_fid_mean):
            axes[1, 0].plot(epochs, val_fid_mean, 'o-', color='red', label='mean')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('FID')
            axes[1, 0].set_title('Validation distribution FID')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Training time
        train_time = [m.get('train_time', None) for m in metrics_history]
        
        if any(v is not None for v in train_time):
            axes[1, 1].plot(epochs, train_time, 'o-', label='Train Time', color='blue')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_title('Training Time per Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved training curves: {save_path}")


_INCEPTION_CACHE = {}


@torch.no_grad()
def _inception_features(images, device, batch_size=64):
    from torchvision.models import inception_v3, Inception_V3_Weights
    import torch.nn.functional as F
    key = str(device)
    if key not in _INCEPTION_CACHE:
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        model.fc = torch.nn.Identity()
        _INCEPTION_CACHE[key] = model.to(device).eval()
    model = _INCEPTION_CACHE[key]
    features = []
    for batch in images.split(batch_size):
        batch = ((batch.to(device) + 1) / 2).clamp(0, 1)
        if batch.size(1) == 1:
            batch = batch.repeat(1, 3, 1, 1)
        batch = F.interpolate(batch, (299, 299), mode='bilinear', align_corners=False)
        mean = batch.new_tensor([.485, .456, .406])[None, :, None, None]
        std = batch.new_tensor([.229, .224, .225])[None, :, None, None]
        features.append(model((batch - mean) / std).cpu())
    return torch.cat(features)


def compute_fid_score(real_images, generated_images, device='cuda', batch_size=64):
    """Distribution FID from torchvision Inception-v3 ImageNet features."""
    from scipy.linalg import sqrtm
    if len(real_images) < 2 or len(generated_images) < 2:
        raise ValueError('FID needs at least two real and two generated images')
    real = _inception_features(real_images, device, batch_size).double().numpy()
    generated = _inception_features(generated_images, device, batch_size).double().numpy()
    difference = real.mean(0) - generated.mean(0)
    cov_real, cov_generated = np.cov(real, rowvar=False), np.cov(generated, rowvar=False)
    covariance_mean = sqrtm(cov_real @ cov_generated).real
    value = difference @ difference + np.trace(cov_real + cov_generated - 2 * covariance_mean)
    return max(float(value), 0.0)


def compute_stats(values):
    """Mean and quartiles of finite, nonnegative per-example measurements."""
    valid = values[torch.isfinite(values) & (values >= 0)].float()
    if valid.numel() == 0:
        return float('nan'), float('nan'), float('nan')
    return valid.mean().item(), valid.quantile(.25).item(), valid.quantile(.75).item()


def compute_mse(
    real_images: torch.Tensor,
    reconstructed_images: torch.Tensor
) -> float:
    """
    Compute Mean Squared Error between images.

    Args:
        real_images: Real images [B, C, H, W]
        reconstructed_images: Reconstructed images [B, C, H, W]

    Returns:
        MSE value
    """
    mse = torch.nn.functional.mse_loss(reconstructed_images, real_images)
    return mse.item()


def compute_per_sample_metrics(real_images, reconstructed_images, device='cuda', batch_size=64):
    """Return pixel MSE and squared Inception feature distance for each pair."""
    mse = torch.nn.functional.mse_loss(reconstructed_images, real_images,
                                      reduction='none').mean(dim=(1, 2, 3))
    real = _inception_features(real_images, device, batch_size)
    reconstructed = _inception_features(reconstructed_images, device, batch_size)
    distance = (real - reconstructed).square().sum(-1).to(mse.device)
    return mse, distance


def compute_topk_metrics(
    real_images: torch.Tensor,
    reconstructed_images: torch.Tensor,
    top_k: int = 50,
    device: str = 'cuda'
) -> Tuple[float, float, float, float, Dict]:
    """
    Compute Top-K metrics: Top-MSE FID and Top-feature-distance MSE.

    Args:
        real_images: Real images [B, C, H, W]
        reconstructed_images: Reconstructed images [B, C, H, W]
        top_k: Number of top samples to select
        device: Device for computation

    Returns:
        Tuple of (top_mse_fid, top_feature_mse, top_mse_best, top_feature_best, detailed_info)
        where detailed_info contains indices and values for analysis
    """
    B = real_images.size(0)

    # Ensure we don't exceed available samples
    top_k = min(top_k, B)

    # Compute per-sample metrics
    per_sample_mse, per_sample_feature_distance = compute_per_sample_metrics(
        real_images, reconstructed_images, device
    )

    # Sort by MSE (ascending - lower MSE is better)
    mse_sorted_indices = torch.argsort(per_sample_mse)
    top_mse_indices = mse_sorted_indices[:top_k]

    # Rank paired feature distances in ascending order.
    feature_sorted_indices = torch.argsort(per_sample_feature_distance)
    top_feature_indices = feature_sorted_indices[:top_k]

    # Compute metrics for top-MSE samples
    top_mse_images_real = real_images[top_mse_indices]
    top_mse_images_recon = reconstructed_images[top_mse_indices]

    # Compute overall FID for top-MSE samples
    top_mse_fid = compute_fid_score(
        top_mse_images_real, top_mse_images_recon, device
    )

    # Compute metrics for top-feature-distance samples
    top_feature_images_real = real_images[top_feature_indices]
    top_feature_images_recon = reconstructed_images[top_feature_indices]

    # Compute overall MSE for top-feature-distance samples
    top_feature_mse = compute_mse(top_feature_images_real, top_feature_images_recon)

    # Get the best single sample metrics
    top_mse_best = per_sample_mse[top_mse_indices[0]].item()  # Best MSE (lowest)
    top_feature_best = per_sample_feature_distance[top_feature_indices[0]].item()  # Smallest feature distance

    # Prepare detailed information for saving
    detailed_info = {
        'top_mse_indices': top_mse_indices.cpu().numpy().tolist(),
        'top_feature_indices': top_feature_indices.cpu().numpy().tolist(),
        'top_mse_values': per_sample_mse[top_mse_indices].cpu().numpy().tolist(),
        'top_feature_values': per_sample_feature_distance[top_feature_indices].cpu().numpy().tolist(),
        'top_mse_feature_values': per_sample_feature_distance[top_mse_indices].cpu().numpy().tolist(),
        'top_feature_mse_values': per_sample_mse[top_feature_indices].cpu().numpy().tolist(),
        'top_mse_best': top_mse_best,
        'top_feature_best': top_feature_best,
    }

    return top_mse_fid, top_feature_mse, top_mse_best, top_feature_best, detailed_info


def save_topk_samples_data(
    detailed_info: Dict,
    save_path: str,
    epoch: int
):
    """
    Save detailed Top-K samples data for future analysis.

    Args:
        detailed_info: Dictionary with detailed Top-K metrics
        save_path: Directory to save the data
        epoch: Current epoch number
    """
    import json
    from pathlib import Path

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed data as JSON
    data_path = save_dir / f"topk_samples_epoch_{epoch:04d}.json"

    with open(data_path, 'w') as f:
        json.dump(detailed_info, f, indent=2)

    print(f"Top-K samples data saved to: {data_path}")


def save_all_samples_metrics(
    per_sample_mse: torch.Tensor,
    per_sample_feature_distance: torch.Tensor,
    save_path: str,
    epoch: int
):
    """
    Save all per-sample metrics (MSE, feature distance) for each epoch.

    Args:
        per_sample_mse: Per-sample MSE values [B]
        per_sample_feature_distance: Per-sample feature distances [B]
        save_path: Directory to save the data
        epoch: Current epoch number
    """
    import json
    from pathlib import Path

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Convert tensors to lists
    mse_list = per_sample_mse.cpu().numpy().tolist()
    feature_list = per_sample_feature_distance.cpu().numpy().tolist()

    # Prepare data
    all_samples_data = {
        'epoch': epoch,
        'num_samples': len(mse_list),
        'per_sample_mse': mse_list,
        'per_sample_feature_distance': feature_list,
        'mean_mse': float(np.mean(mse_list)),
        'mean_feature_distance': float(np.mean([f for f in feature_list if f >= 0])),  # Exclude failed FID (-1)
        'std_mse': float(np.std(mse_list)),
        'std_feature_distance': float(np.std([f for f in feature_list if f >= 0])),
        'min_mse': float(np.min(mse_list)),
        'max_mse': float(np.max(mse_list)),
        'min_feature_distance': float(np.min([f for f in feature_list if f >= 0])) if any(f >= 0 for f in feature_list) else -1.0,
        'max_feature_distance': float(np.max([f for f in feature_list if f >= 0])) if any(f >= 0 for f in feature_list) else -1.0,
    }

    # Save as JSON
    data_path = save_dir / f"all_samples_epoch_{epoch:04d}.json"

    with open(data_path, 'w') as f:
        json.dump(all_samples_data, f, indent=2)

    print(f"All samples metrics saved to: {data_path}")
