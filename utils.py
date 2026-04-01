"""
Utility functions for DAMCHA Transformer training and evaluation.
"""

import torch
import torch.nn as nn
import random
import numpy as np
import time
from typing import Tuple


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Patchify(nn.Module):
    """
    Convert images to patch sequences with linear projection.
    
    Supports image_size divisible by patch_size.
    Output shape: [B, T, d_model] where T = (H/patch) * (W/patch)
    """
    
    def __init__(
        self,
        in_ch: int = 3,
        img_size: int = 32,
        patch: int = 4,
        d_model: int = 256
    ):
        super().__init__()
        assert img_size % patch == 0, "img_size must be divisible by patch"
        
        self.in_ch = in_ch
        self.img_size = img_size
        self.patch = patch
        self.grid = img_size // patch
        self.d_model = d_model
        self.num_patches = self.grid * self.grid
        
        self.proj = nn.Linear(in_ch * patch * patch, d_model)
    
    def to_sequence(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patch sequences.
        
        Args:
            imgs: Images [B, C, H, W]
            
        Returns:
            Patch sequences [B, T, d_model]
        """
        B, C, H, W = imgs.shape
        assert C == self.in_ch and H == self.img_size and W == self.img_size
        
        p = self.patch
        
        # Unfold into patches
        patches = imgs.unfold(2, p, p).unfold(3, p, p)  # [B, C, grid, grid, p, p]
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # [B, grid, grid, C, p, p]
        patches = patches.view(B, self.grid * self.grid, C * p * p)  # [B, T, C*p*p]
        
        # Project to d_model
        x = self.proj(patches)  # [B, T, d_model]
        
        return x
    
    def from_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct images from patch sequences.
        
        Args:
            seq: Patch sequences [B, T, d_model]
            
        Returns:
            Reconstructed images [B, C, H, W]
        """
        B, T, _ = seq.shape
        assert T == self.num_patches
        
        # Inverse projection
        patches = self.proj.weight.t() @ seq.transpose(1, 2)  # Approximate inverse
        patches = patches.transpose(1, 2)  # [B, T, C*p*p]
        
        p = self.patch
        patches = patches.view(B, self.grid, self.grid, self.in_ch, p, p)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        imgs = patches.view(B, self.in_ch, self.img_size, self.img_size)
        
        return imgs


class MaskGenerator:
    """
    Generate random masks for self-supervised learning.
    
    Supports row-wise masking: randomly select 1-3 consecutive rows of patches
    to mask (where rows correspond to spatial rows in the patch grid).
    """
    
    def __init__(
        self,
        mask_ratio: float = 0.15,
        grid_size: int = 8,
        min_rows: int = 1,
        max_rows: int = 3,
        mode: str = 'row'
    ):
        """
        Initialize mask generator.
        
        Args:
            mask_ratio: Ratio of tokens to mask (used in 'random' mode)
            grid_size: Number of patches per row/column (e.g., 8 for 32x32 with patch_size=4)
            min_rows: Minimum number of consecutive rows to mask
            max_rows: Maximum number of consecutive rows to mask
            mode: 'row' for row-wise masking, 'random' for random token masking
        """
        self.mask_ratio = mask_ratio
        self.grid_size = grid_size
        self.min_rows = min_rows
        self.max_rows = max_rows
        self.mode = mode
    
    def generate_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: str = 'cuda'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate mask for sequences.
        
        In 'row' mode: randomly select 1-3 consecutive rows of patches to mask.
        In 'random' mode: randomly select individual tokens to mask.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length (number of patches, should be grid_size^2)
            device: Device
            
        Returns:
            Tuple of (mask_indices, mask_bool)
            - mask_indices: Indices of masked positions [B, num_masked]
            - mask_bool: Boolean mask [B, seq_len]
        """
        if self.mode == 'row':
            return self._generate_row_mask(batch_size, seq_len, device)
        else:
            return self._generate_random_mask(batch_size, seq_len, device)
    
    def _generate_row_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate row-wise mask: mask 1-3 consecutive rows of patches."""
        grid = self.grid_size
        assert seq_len == grid * grid, f"seq_len {seq_len} != grid^2 {grid*grid}"
        
        mask_bool = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        all_indices = []
        max_masked = 0
        
        for i in range(batch_size):
            # Randomly choose number of rows to mask (1 to max_rows)
            num_rows = random.randint(self.min_rows, min(self.max_rows, grid))
            
            # Randomly choose starting row
            start_row = random.randint(0, grid - num_rows)
            
            # Get indices of all patches in these rows
            indices = []
            for row in range(start_row, start_row + num_rows):
                row_start = row * grid
                row_end = row_start + grid
                indices.extend(range(row_start, row_end))
            
            indices = torch.tensor(indices, device=device)
            mask_bool[i, indices] = True
            all_indices.append(indices)
            max_masked = max(max_masked, len(indices))
        
        # Pad indices to same length
        padded_indices = torch.zeros(batch_size, max_masked, dtype=torch.long, device=device)
        for i, indices in enumerate(all_indices):
            padded_indices[i, :len(indices)] = indices
        
        return padded_indices, mask_bool
    
    def _generate_random_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random token mask (original behavior)."""
        num_masked = max(1, int(seq_len * self.mask_ratio))
        
        mask_bool = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        mask_indices = []
        
        for i in range(batch_size):
            indices = torch.randperm(seq_len, device=device)[:num_masked]
            mask_bool[i, indices] = True
            mask_indices.append(indices)
        
        mask_indices = torch.stack(mask_indices)
        
        return mask_indices, mask_bool
    
    def apply_mask(
        self,
        x: torch.Tensor,
        mask_bool: torch.Tensor,
        mask_token: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply row-wise mask to input sequences.
        
        Masked rows are replaced with mask_token (or zeros).
        
        Args:
            x: Input sequences [B, T, d_model]
            mask_bool: Boolean mask [B, T] indicating which rows to mask
            mask_token: Token to replace masked rows (if None, use zeros)
            
        Returns:
            Masked sequences [B, T, d_model]
        """
        B, T, d_model = x.shape
        
        if mask_token is None:
            mask_token = torch.zeros(d_model, device=x.device)
        
        x_masked = x.clone()
        # mask_bool[i, j] = True means row j of sample i should be masked
        x_masked[mask_bool] = mask_token
        
        return x_masked


class AverageMeter:
    """Compute and store average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    """Simple timer for measuring elapsed time."""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        if self.start_time is not None:
            self.elapsed = time.time() - self.start_time
            self.start_time = None
        return self.elapsed
    
    def get_elapsed(self):
        if self.start_time is not None:
            return time.time() - self.start_time
        return self.elapsed


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    loss: float,
    path: str,
    save_off_diag_params: bool = True,
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        path: Path to save checkpoint
        save_off_diag_params: Whether to save off-diagonal parameters separately
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # Save off-diagonal parameters separately if available
    if save_off_diag_params and hasattr(model, 'get_off_diag_params'):
        off_diag_params = model.get_off_diag_params()
        checkpoint['off_diag_params'] = off_diag_params
        
        # Also save to a separate file for easy inspection
        import os
        off_diag_path = path.replace('.pt', '_off_diag_params.pt')
        torch.save(off_diag_params, off_diag_path)
    
    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    optimizer,
    path: str,
    device: str = 'cuda'
):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
