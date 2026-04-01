"""
Training script for DAMCHA Transformer on NLP tasks.

Supports:
- WMT (Machine Translation)
- CNN/DailyMail (Summarization)
- CommonGen (Concept-to-Text Generation)

Training modes:
- Standard Transformer (default)
- M0-only attention (--use_M)
- MLP-based M generation (--use_M --use_mlp)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from typing import Tuple, Optional, Dict, List

from models.transformer_damcha import TransformerPC
from models.baselines import BaselineTransformer, BASELINE_NAMES
from utils import AverageMeter, Timer, count_parameters, get_lr, save_checkpoint
from visualization import MetricsLogger, Visualizer
from evaluation_nlp import (
    compute_all_metrics, compute_topk_nlp_metrics, NLPMetricsLogger,
    compute_per_sample_nlp_metrics, save_all_samples_nlp_metrics,
    compute_per_sample_bartscore,
)


class Seq2SeqHead(nn.Module):
    """
    Sequence-to-sequence head for NLP tasks.
    Maps decoder output to vocabulary logits.
    """
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return self.proj(x)


class NLPTrainer:
    """Trainer for DAMCHA Transformer on NLP tasks."""
    
    def __init__(
        self,
        model: TransformerPC,
        head: Seq2SeqHead,
        tokenizer,
        device: str = 'cuda',
        dev_mode: bool = False,
        task: str = 'machine_translation',
        top_k_samples: int = 50,
        compute_topk_metrics: bool = False,
        metrics_save_dir: str = './outputs/metrics',
    ):
        self.model = model
        self.head = head
        self.tokenizer = tokenizer
        self.device = device
        self.dev_mode = dev_mode
        self.task = task
        self.top_k_samples = top_k_samples
        self.compute_topk_metrics_flag = compute_topk_metrics
        self.metrics_save_dir = metrics_save_dir
        
        # Move to device
        self.model.to(device)
        self.head.to(device)
        
        # Loss function (ignore padding token)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
        )
    
    def _compute_loss(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.
        
        Args:
            logits: Model output logits [B, T, vocab_size]
            target_ids: Target token IDs [B, T]
            target_mask: Target attention mask [B, T]
            
        Returns:
            Scalar loss
        """
        # Shift for autoregressive prediction
        # logits: predict next token, target: actual next token
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()
        
        # Flatten for loss computation
        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return loss
    
    def train_epoch(
        self,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        epoch: int,
    ) -> Tuple[float, float]:
        """
        Train one epoch.
        
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
        
        for batch_idx, batch in enumerate(loader):
            source_ids = batch['source_ids'].to(self.device)
            source_mask = batch['source_mask'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            target_mask = batch['target_mask'].to(self.device)
            
            B = source_ids.size(0)
            
            optimizer.zero_grad()
            
            # For decoder-only model, concatenate source and target
            # Input: [source_ids, target_ids[:-1]]
            # Target: [source_ids (ignored), target_ids[1:]]
            input_ids = target_ids  # Use target for autoregressive training
            
            # Create embeddings (simple embedding layer)
            # Note: In full implementation, would use proper embedding
            # Here we use the model directly with token IDs converted to embeddings
            
            # Forward pass through decoder
            # The model expects [B, T, d_model] input
            # We need an embedding layer
            dec_out = self.model(self._embed_tokens(input_ids))
            
            # Get logits
            logits = self.head(dec_out)
            
            # Compute loss
            loss = self._compute_loss(logits, target_ids, target_mask)
            
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
        
        epoch_time = timer.stop()
        return loss_meter.avg, epoch_time
    
    def _embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings.
        Uses a simple learned embedding.
        """
        if not hasattr(self, 'embedding'):
            vocab_size = self.tokenizer.vocab_size
            d_model = self.model.d_model
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.tokenizer.pad_token_id)
            self.embedding.to(self.device)
        
        return self.embedding(token_ids)
    
    @torch.no_grad()
    def generate(
        self,
        source_ids: torch.Tensor,
        max_length: int = 128,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            source_ids: Source token IDs [B, T_src]
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated token IDs [B, T_gen]
        """
        self.model.eval()
        self.head.eval()
        
        B = source_ids.size(0)
        device = source_ids.device
        
        # Start with BOS token
        if self.tokenizer.bos_token_id is not None:
            generated = torch.full((B, 1), self.tokenizer.bos_token_id, device=device)
        else:
            generated = torch.full((B, 1), self.tokenizer.pad_token_id, device=device)
        
        for _ in range(max_length - 1):
            # Get embeddings
            embeddings = self._embed_tokens(generated)
            
            # Forward pass
            dec_out = self.model(embeddings)
            
            # Get logits for last position
            logits = self.head(dec_out[:, -1:, :])  # [B, 1, vocab_size]
            logits = logits / temperature
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1)  # Greedy decoding
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if self.tokenizer.eos_token_id is not None:
                if (next_token == self.tokenizer.eos_token_id).all():
                    break
        
        return generated
    
    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        epoch: int = 0,
        compute_metrics: bool = True,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate model on validation set.
        
        Args:
            loader: Validation data loader
            epoch: Current epoch
            compute_metrics: Whether to compute NLP metrics
            
        Returns:
            Tuple of (val_loss, metrics_dict)
        """
        self.model.eval()
        self.head.eval()
        
        loss_meter = AverageMeter()
        all_hypotheses = []
        all_references = []
        all_per_sample_nll = []  # Per-sample NLL (negative log-likelihood)
        all_per_sample_ppl = []  # Per-sample perplexity
        
        # Limit validation batches for speed
        MAX_EVAL_BATCHES = 50
        
        for batch_idx, batch in enumerate(loader):
            
            source_ids = batch['source_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            target_mask = batch['target_mask'].to(self.device)
            source_texts = batch['source_texts']
            target_texts = batch['target_texts']
            
            B = source_ids.size(0)
            
            # Compute loss
            input_ids = target_ids
            embeddings = self._embed_tokens(input_ids)
            dec_out = self.model(embeddings)
            logits = self.head(dec_out)
            loss = self._compute_loss(logits, target_ids, target_mask)
            
            loss_meter.update(loss.item(), B)
            
            # Compute per-sample NLL and perplexity
            if compute_metrics:
                # Shift for autoregressive prediction
                shift_logits = logits[:, :-1, :].contiguous()  # [B, T-1, vocab_size]
                shift_labels = target_ids[:, 1:].contiguous()  # [B, T-1]
                
                # Compute per-sample cross-entropy loss (NLL)
                for i in range(B):
                    sample_logits = shift_logits[i]  # [T-1, vocab_size]
                    sample_labels = shift_labels[i]  # [T-1]
                    
                    # Mask out padding tokens
                    pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else -100
                    valid_mask = sample_labels != pad_id
                    
                    if valid_mask.sum() > 0:
                        valid_logits = sample_logits[valid_mask]
                        valid_labels = sample_labels[valid_mask]
                        sample_nll = F.cross_entropy(valid_logits, valid_labels).item()
                        sample_ppl = float(torch.exp(torch.tensor(sample_nll)))
                    else:
                        sample_nll = -1.0
                        sample_ppl = -1.0
                    
                    all_per_sample_nll.append(sample_nll)
                    all_per_sample_ppl.append(sample_ppl)
                
                # Generate for text metrics
                generated_ids = self.generate(source_ids, max_length=target_ids.size(1))
                
                # Decode generated text
                for i in range(B):
                    gen_text = self.tokenizer.decode(
                        generated_ids[i], skip_special_tokens=True
                    )
                    all_hypotheses.append(gen_text)
                    all_references.append(target_texts[i])
            
            if batch_idx >= MAX_EVAL_BATCHES - 1:
                break
        
        val_loss = loss_meter.avg
        metrics = {'val_loss': val_loss}
        
        # Compute NLP metrics
        if compute_metrics and len(all_hypotheses) > 0:
            # Compute per-sample BARTScore
            print("Computing per-sample BARTScore...")
            all_per_sample_bartscore = compute_per_sample_bartscore(
                all_hypotheses, all_references,
                device=self.device, batch_size=8
            )
            
            # Aggregate: NLL/BARTScore mean/Q25/Q75 + top-50
            nlp_metrics = compute_all_metrics(
                all_hypotheses, all_references,
                per_sample_nll=all_per_sample_nll,
                per_sample_bartscore=all_per_sample_bartscore,
                task=self.task, device=self.device,
                top_k=self.top_k_samples,
            )
            metrics.update(nlp_metrics)
            
            # Save all per-sample data to JSON
            per_sample_metrics = compute_per_sample_nlp_metrics(
                all_hypotheses, all_references, self.task
            )
            per_sample_metrics['per_sample_nll'] = all_per_sample_nll
            per_sample_metrics['per_sample_ppl'] = all_per_sample_ppl
            per_sample_metrics['per_sample_bartscore'] = all_per_sample_bartscore
            save_all_samples_nlp_metrics(
                all_hypotheses, all_references, per_sample_metrics,
                self.metrics_save_dir, epoch
            )
        
        return val_loss, metrics


def build_nlp_model(
    d_model: int,
    n_heads: int,
    d_ff: int,
    n_layers: int,
    use_M: bool,
    use_mlp: bool,
    share_mlp: bool,
    mlp_hidden: Tuple[int, ...],
    dev_mode: bool,
    device: str,
    off_diag_mode: str = 'mlp',
    diag_scale: float = 1.0,
    off_diag_scale: float = 0.5,
    off_diag_alpha_init: float = 0.1,
    use_layer_bias: bool = False,
    layer_bias_rank: int = 16,
    use_multihead_M: bool = False,
    baseline: str = None,
    compact_rank: int = 0,
    free_M: bool = False,
):
    """
    Build decoder-only model for NLP tasks.

    When baseline is set, a BaselineTransformer is returned instead of
    TransformerPC and all M-related parameters are ignored.
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


def run_block_ablation_evaluation_nlp(
    model: TransformerPC,
    trainer: 'NLPTrainer',
    val_loader: DataLoader,
    config: dict,
    metrics_dir: str,
):
    """
    Run block ablation evaluation for NLP tasks with diagonal-only and off-diagonal-only M matrix.
    
    Args:
        model: Trained TransformerPC model
        trainer: NLPTrainer instance
        val_loader: Validation data loader
        config: Training configuration
        metrics_dir: Directory to save metrics
    """
    import json
    
    # Store original scales
    original_diag_scale = config.get('diag_scale', 1.0)
    original_off_diag_scale = config.get('off_diag_scale', 0.5)
    
    results = {}
    
    # Evaluate with diagonal-only (diag_scale=1.0, off_diag_scale=0.0)
    print("\n[Diagonal-only] Evaluating with diag_scale=1.0, off_diag_scale=0.0...")
    modify_model_scales(model, diag_scale=1.0, off_diag_scale=0.0)
    
    diag_loss, diag_metrics = trainer.evaluate(val_loader, epoch=-1, compute_metrics=True)
    
    results['diagonal_only'] = {
        'val_loss': float(diag_loss),
        'bartscore': float(diag_metrics.get('bartscore', -1.0)),
        'top50_bartscore': float(diag_metrics.get('top50_bartscore', -1.0)),
        'bleu': float(diag_metrics.get('bleu', -1.0)),
    }
    print(f"  bartscore: {diag_metrics.get('bartscore', -1.0):.4f}, top50_bartscore: {diag_metrics.get('top50_bartscore', -1.0):.4f}")
    
    # Evaluate with off-diagonal-only (diag_scale=0.0, off_diag_scale=1.0)
    print("\n[Off-diagonal-only] Evaluating with diag_scale=0.0, off_diag_scale=1.0...")
    modify_model_scales(model, diag_scale=0.0, off_diag_scale=1.0)
    
    offdiag_loss, offdiag_metrics = trainer.evaluate(val_loader, epoch=-1, compute_metrics=True)
    
    results['offdiag_only'] = {
        'val_loss': float(offdiag_loss),
        'bartscore': float(offdiag_metrics.get('bartscore', -1.0)),
        'top50_bartscore': float(offdiag_metrics.get('top50_bartscore', -1.0)),
        'bleu': float(offdiag_metrics.get('bleu', -1.0)),
    }
    print(f"  bartscore: {offdiag_metrics.get('bartscore', -1.0):.4f}, top50_bartscore: {offdiag_metrics.get('top50_bartscore', -1.0):.4f}")
    
    # Restore original scales
    modify_model_scales(model, diag_scale=original_diag_scale, off_diag_scale=original_off_diag_scale)
    
    # Save results
    results_file = os.path.join(metrics_dir, 'block_ablation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBlock ablation results saved to: {results_file}")
    print("="*80)


def train_nlp(
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer,
    config: dict,
):
    """
    Main training function for NLP tasks.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        tokenizer: HuggingFace tokenizer
        config: Configuration dictionary
    """
    device = config['device']
    dev_mode = config['dev_mode']
    dataset_name = config.get('dataset', 'commongen')
    
    # Get task type
    task_map = {
        'wmt': 'machine_translation',
        'cnn_dailymail': 'summarization',
        'commongen': 'concept_to_text',
    }
    task = task_map.get(dataset_name.lower(), 'concept_to_text')
    
    # Determine mode name for output directories
    baseline = config.get('baseline')
    if baseline and baseline in BASELINE_NAMES:
        mode_name = f'baselines/{baseline}'
    elif not config['use_M']:
        mode_name = 'standard'
    elif config['use_mlp']:
        # Add MLP depth info to mode name
        mlp_hidden = config.get('mlp_hidden', ())
        mlp_depth = len(mlp_hidden) + 1  # +1 for output layer
        depth_suffix = f'_depth{mlp_depth}'
        
        # Add multihead_M suffix if enabled
        multihead_suffix = '_multihead' if config.get('use_multihead_M', False) else ''
        cr = config.get('compact_rank', 0)
        compact_suffix = f'_compact{cr}' if cr > 0 else ''
        free_suffix = '_freeM' if config.get('free_M', False) else ''

        if config['share_mlp']:
            mode_name = f'mlp_shared{depth_suffix}{multihead_suffix}{compact_suffix}{free_suffix}'
        else:
            mode_name = f'mlp_separate{depth_suffix}{multihead_suffix}{compact_suffix}{free_suffix}'
    else:
        mode_name = 'M0_only'
    
    # Initialize output directories
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_mode = mode_name.replace('/', '_')
    experiment_name = f"{safe_mode}_{timestamp}"
    
    output_base = config.get('output_dir', './outputs')
    vis_dir = os.path.join(output_base, dataset_name, 'visualizations', mode_name)
    metrics_dir = os.path.join(output_base, dataset_name, 'metrics', mode_name)
    save_dir = os.path.join(config.get('save_dir', './checkpoints'), dataset_name, mode_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_logger = NLPMetricsLogger(
        save_dir=metrics_dir,
        experiment_name=experiment_name
    )
    
    print(f"\n[Output Directories]")
    print(f"  Dataset: {dataset_name}")
    print(f"  Task: {task}")
    print(f"  Mode: {mode_name}")
    print(f"  Checkpoints: {save_dir}")
    print(f"  Metrics: {metrics_dir}\n")
    
    # Build model
    model = build_nlp_model(
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        n_layers=config['n_layers'],
        use_M=config['use_M'],
        use_mlp=config['use_mlp'],
        share_mlp=config['share_mlp'],
        mlp_hidden=config['mlp_hidden'],
        dev_mode=dev_mode,
        device=device,
        off_diag_mode=config.get('off_diag_mode', 'mlp'),
        diag_scale=config.get('diag_scale', 1.0),
        off_diag_scale=config.get('off_diag_scale', 0.5),
        off_diag_alpha_init=config.get('off_diag_alpha_init', 0.1),
        use_layer_bias=config.get('use_layer_bias', False),
        layer_bias_rank=config.get('layer_bias_rank', 16),
        use_multihead_M=config.get('use_multihead_M', False),
        baseline=config.get('baseline'),
        compact_rank=config.get('compact_rank', 0),
        free_M=config.get('free_M', False),
    )
    
    # Build head
    vocab_size = tokenizer.vocab_size
    head = Seq2SeqHead(config['d_model'], vocab_size).to(device)
    
    # Print model info
    total_params = count_parameters(model) + count_parameters(head)
    print(f"\nModel initialized:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Device: {device}")
    print(f"  Dev mode: {dev_mode}\n")
    
    # Trainer
    trainer = NLPTrainer(
        model, head, tokenizer,
        device=device,
        dev_mode=dev_mode,
        task=task,
        top_k_samples=config.get('top_k_samples', 50),
        compute_topk_metrics=config.get('compute_topk_metrics', False),
        metrics_save_dir=metrics_dir,
    )
    
    # Optimizer (include embedding layer)
    all_params = list(model.parameters()) + list(head.parameters())
    if hasattr(trainer, 'embedding'):
        all_params += list(trainer.embedding.parameters())
    
    optimizer = optim.AdamW(
        all_params,
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['lr'] * 0.01
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_time = trainer.train_epoch(
            train_loader, optimizer, epoch
        )
        
        print(f"\nTraining completed - Loss: {train_loss:.4f}, Time: {train_time:.2f}s")
        
        # Validation
        val_loss, metrics = trainer.evaluate(
            val_loader, epoch=epoch, compute_metrics=True
        )
        
        print(f"Validation - Loss: {val_loss:.4f}")
        print(f"  NLL      : mean={metrics.get('nll_mean',-1):.4f}  Q25={metrics.get('nll_q25',-1):.4f}  Q75={metrics.get('nll_q75',-1):.4f}  top50={metrics.get('top50_nll_mean',-1):.4f}")
        print(f"  BARTScore: mean={metrics.get('bartscore_mean',-1):.4f}  Q25={metrics.get('bartscore_q25',-1):.4f}  Q75={metrics.get('bartscore_q75',-1):.4f}  top50={metrics.get('top50_bartscore_mean',-1):.4f}")
        
        # Log metrics
        metrics_dict = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_time': train_time,
            'learning_rate': get_lr(optimizer),
            **metrics
        }
        metrics_logger.log_epoch(metrics_dict)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(save_dir, 'best_model.pt')
            )
            print(f"Best model saved (val_loss: {val_loss:.4f})")
        
        # Save regular checkpoint
        if epoch % config.get('save_every', 10) == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
            )
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
    print(f"Metrics saved to: {metrics_logger.csv_path}")
    
    # Run block ablation evaluation if enabled
    if config.get('run_block_ablation', False) and config.get('use_M', False) and config.get('use_mlp', False):
        print("\n" + "="*80)
        print("Running Block Ablation Evaluation")
        print("="*80)
        
        run_block_ablation_evaluation_nlp(
            model=model,
            trainer=trainer,
            val_loader=val_loader,
            config=config,
            metrics_dir=metrics_dir,
        )
