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
    
    def __init__(self, save_dir: str, experiment_name: str = None):
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
            'val_fid_mean', 'val_fid_q25', 'val_fid_q75',
            'learning_rate', 'timestamp',
            'top_mse_fid', 'top_fid_mse', 'top_mse_best', 'top_fid_best'
        ]
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()
        
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
        val_fid_q25  = [m.get('val_fid_q25',  None) for m in metrics_history]
        val_fid_q75  = [m.get('val_fid_q75',  None) for m in metrics_history]
        if any(v is not None for v in val_fid_mean):
            axes[1, 0].plot(epochs, val_fid_mean, 'o-', color='red', label='mean')
            if any(v is not None for v in val_fid_q25):
                axes[1, 0].fill_between(epochs, val_fid_q25, val_fid_q75,
                                        alpha=0.25, color='red', label='Q25-Q75')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('FID')
            axes[1, 0].set_title('Validation FID (mean ± IQR)')
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


def compute_fid_score(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    device: str = 'cuda',
    batch_size: int = 64
) -> float:
    """
    Compute Frechet Inception Distance (FID) using InceptionV3 features.
    
    Uses torchvision's pretrained InceptionV3 to extract features,
    then computes FID between real and generated image distributions.
    
    Args:
        real_images: Real images [B, C, H, W] (normalized to [-1, 1])
        generated_images: Generated images [B, C, H, W] (normalized to [-1, 1])
        device: Device for computation
        batch_size: Batch size for feature extraction
        
    Returns:
        FID score (lower is better), or -1.0 if computation fails
    """
    try:
        from scipy import linalg
        from torchvision import models
        import torch.nn.functional as F
        
        # Load InceptionV3 pretrained model
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False)
        inception.fc = torch.nn.Identity()  # Remove final FC layer to get features
        inception = inception.to(device)
        inception.eval()
        
        def get_inception_features(images: torch.Tensor) -> np.ndarray:
            """Extract InceptionV3 features from images."""
            # Move to device first
            images = images.to(device)
            
            # Denormalize from [-1, 1] to [0, 1]
            images = (images + 1) / 2
            images = images.clamp(0, 1)
            
            # Resize to InceptionV3 input size (299x299)
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
            
            # Normalize with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            images = (images - mean) / std
            
            features = []
            with torch.no_grad():
                for i in range(0, len(images), batch_size):
                    batch = images[i:i+batch_size]
                    feat = inception(batch)
                    features.append(feat.cpu().numpy())
            
            return np.concatenate(features, axis=0)
        
        # Extract features
        real_features = get_inception_features(real_images)
        gen_features = get_inception_features(generated_images)
        
        # Limit samples for efficiency
        max_samples = 1000
        if real_features.shape[0] > max_samples:
            idx = np.random.choice(real_features.shape[0], max_samples, replace=False)
            real_features = real_features[idx]
            gen_features = gen_features[idx]
        
        # Compute statistics
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        
        mu_gen = np.mean(gen_features, axis=0)
        sigma_gen = np.cov(gen_features, rowvar=False)
        
        # Ensure covariance matrices are 2D
        if sigma_real.ndim == 0:
            sigma_real = np.array([[sigma_real]])
        if sigma_gen.ndim == 0:
            sigma_gen = np.array([[sigma_gen]])
        
        # Compute FID: ||mu_real - mu_gen||^2 + Tr(sigma_real + sigma_gen - 2*sqrt(sigma_real*sigma_gen))
        diff = mu_real - mu_gen
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
        
        return float(fid)
    
    except Exception as e:
        print(f"Warning: FID computation failed: {e}")
        return -1.0


def compute_stats(values: torch.Tensor) -> Tuple[float, float, float]:
    """
    Compute mean, Q25, Q75 from a 1D tensor of per-sample values.
    Ignores negative sentinel values (-1.0).

    Returns:
        Tuple of (mean, q25, q75), each -1.0 on failure.
    """
    try:
        valid = values[values >= 0].float()
        if len(valid) == 0:
            return -1.0, -1.0, -1.0
        mean = valid.mean().item()
        q25 = torch.quantile(valid, 0.25).item()
        q75 = torch.quantile(valid, 0.75).item()
        return mean, q25, q75
    except Exception:
        return -1.0, -1.0, -1.0


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


def compute_per_sample_metrics(
    real_images: torch.Tensor,
    reconstructed_images: torch.Tensor,
    device: str = 'cuda',
    batch_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute MSE and FID for each sample individually.

    Args:
        real_images: Real images [B, C, H, W]
        reconstructed_images: Reconstructed images [B, C, H, W]
        device: Device for computation
        batch_size: Batch size for FID computation

    Returns:
        Tuple of (per_sample_mse, per_sample_fid)
    """
    B = real_images.size(0)

    # Compute per-sample MSE
    per_sample_mse = torch.nn.functional.mse_loss(
        reconstructed_images, real_images, reduction='none'
    ).mean(dim=[1, 2, 3])  # [B]

    # For FID, we need to compute features for each sample
    per_sample_fid = torch.full((B,), -1.0, device=real_images.device)

    try:
        from torchvision.models import inception_v3
        from scipy import linalg
        import numpy as np

        # Load InceptionV3 for feature extraction
        from torchvision.models import Inception_V3_Weights
        inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        inception.eval()
        inception.fc = torch.nn.Identity()  # Remove final classification layer
        inception.to(device)

        # Process images in batches for FID computation
        for i in range(0, B, batch_size):
            batch_end = min(i + batch_size, B)
            batch_real = real_images[i:batch_end].to(device)
            batch_recon = reconstructed_images[i:batch_end].to(device)

            # Normalize to [0, 1] for InceptionV3
            batch_real = (batch_real + 1.0) / 2.0
            batch_recon = (batch_recon + 1.0) / 2.0

            # Resize to 299x299 for InceptionV3
            batch_real = torch.nn.functional.interpolate(
                batch_real, size=(299, 299), mode='bilinear', align_corners=False
            )
            batch_recon = torch.nn.functional.interpolate(
                batch_recon, size=(299, 299), mode='bilinear', align_corners=False
            )

            with torch.no_grad():
                # Extract features
                features_real = inception(batch_real).cpu().numpy()  # [batch, 2048]
                features_recon = inception(batch_recon).cpu().numpy()

                # Compute FID for each sample in the batch
                for j in range(features_real.shape[0]):
                    mu_real = features_real[j:j+1].mean(axis=0)
                    mu_recon = features_recon[j:j+1].mean(axis=0)

                    # For single samples, we treat the sample as its own distribution
                    # This is a simplified FID approximation
                    fid = np.sum((mu_real - mu_recon) ** 2)
                    per_sample_fid[i + j] = float(fid)

    except Exception as e:
        print(f"Warning: Per-sample FID computation failed: {e}")
        # Keep -1.0 values for per_sample_fid

    return per_sample_mse, per_sample_fid


def compute_topk_metrics(
    real_images: torch.Tensor,
    reconstructed_images: torch.Tensor,
    top_k: int = 50,
    device: str = 'cuda'
) -> Tuple[float, float, float, float, Dict]:
    """
    Compute Top-K metrics: Top-MSE FID and Top-FID MSE.

    Args:
        real_images: Real images [B, C, H, W]
        reconstructed_images: Reconstructed images [B, C, H, W]
        top_k: Number of top samples to select
        device: Device for computation

    Returns:
        Tuple of (top_mse_fid, top_fid_mse, top_mse_best, top_fid_best, detailed_info)
        where detailed_info contains indices and values for analysis
    """
    B = real_images.size(0)

    # Ensure we don't exceed available samples
    top_k = min(top_k, B)

    # Compute per-sample metrics
    per_sample_mse, per_sample_fid = compute_per_sample_metrics(
        real_images, reconstructed_images, device
    )

    # Sort by MSE (ascending - lower MSE is better)
    mse_sorted_indices = torch.argsort(per_sample_mse)
    top_mse_indices = mse_sorted_indices[:top_k]

    # Sort by FID (ascending - lower FID is better)
    fid_sorted_indices = torch.argsort(per_sample_fid)
    top_fid_indices = fid_sorted_indices[:top_k]

    # Compute metrics for top-MSE samples
    top_mse_images_real = real_images[top_mse_indices]
    top_mse_images_recon = reconstructed_images[top_mse_indices]

    # Compute overall FID for top-MSE samples
    top_mse_fid = compute_fid_score(
        top_mse_images_real, top_mse_images_recon, device
    )

    # Compute metrics for top-FID samples
    top_fid_images_real = real_images[top_fid_indices]
    top_fid_images_recon = reconstructed_images[top_fid_indices]

    # Compute overall MSE for top-FID samples
    top_fid_mse = compute_mse(top_fid_images_real, top_fid_images_recon)

    # Get the best single sample metrics
    top_mse_best = per_sample_mse[top_mse_indices[0]].item()  # Best MSE (lowest)
    top_fid_best = per_sample_fid[top_fid_indices[0]].item()  # Best FID (lowest)

    # Prepare detailed information for saving
    detailed_info = {
        'top_mse_indices': top_mse_indices.cpu().numpy().tolist(),
        'top_fid_indices': top_fid_indices.cpu().numpy().tolist(),
        'top_mse_values': per_sample_mse[top_mse_indices].cpu().numpy().tolist(),
        'top_fid_values': per_sample_fid[top_fid_indices].cpu().numpy().tolist(),
        'top_mse_fid_values': per_sample_fid[top_mse_indices].cpu().numpy().tolist(),
        'top_fid_mse_values': per_sample_mse[top_fid_indices].cpu().numpy().tolist(),
        'top_mse_best': top_mse_best,
        'top_fid_best': top_fid_best,
    }

    return top_mse_fid, top_fid_mse, top_mse_best, top_fid_best, detailed_info


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
    per_sample_fid: torch.Tensor,
    save_path: str,
    epoch: int
):
    """
    Save all per-sample metrics (MSE, FID) for each epoch.

    Args:
        per_sample_mse: Per-sample MSE values [B]
        per_sample_fid: Per-sample FID values [B]
        save_path: Directory to save the data
        epoch: Current epoch number
    """
    import json
    from pathlib import Path

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Convert tensors to lists
    mse_list = per_sample_mse.cpu().numpy().tolist()
    fid_list = per_sample_fid.cpu().numpy().tolist()

    # Prepare data
    all_samples_data = {
        'epoch': epoch,
        'num_samples': len(mse_list),
        'per_sample_mse': mse_list,
        'per_sample_fid': fid_list,
        'mean_mse': float(np.mean(mse_list)),
        'mean_fid': float(np.mean([f for f in fid_list if f >= 0])),  # Exclude failed FID (-1)
        'std_mse': float(np.std(mse_list)),
        'std_fid': float(np.std([f for f in fid_list if f >= 0])),
        'min_mse': float(np.min(mse_list)),
        'max_mse': float(np.max(mse_list)),
        'min_fid': float(np.min([f for f in fid_list if f >= 0])) if any(f >= 0 for f in fid_list) else -1.0,
        'max_fid': float(np.max([f for f in fid_list if f >= 0])) if any(f >= 0 for f in fid_list) else -1.0,
    }

    # Save as JSON
    data_path = save_dir / f"all_samples_epoch_{epoch:04d}.json"

    with open(data_path, 'w') as f:
        json.dump(all_samples_data, f, indent=2)

    print(f"All samples metrics saved to: {data_path}")
