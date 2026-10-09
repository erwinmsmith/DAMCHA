"""
NLP evaluation utilities for DAMCHA experiments.

Metrics:
  CV  : MSE (mean/Q25/Q75), FID (mean/Q25/Q75), top-50
  NLP : NLL (mean/Q25/Q75), BARTScore (mean/Q25/Q75), top-50 NLL, top-50 BARTScore
"""

import os
import csv
import json
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch


# ---------------------------------------------------------------------------
# Metrics Logger
# ---------------------------------------------------------------------------

class NLPMetricsLogger:
    """CSV metrics logger for NLP training."""

    HEADERS = [
        'epoch', 'train_loss', 'train_time', 'val_loss',
        'nll_mean', 'nll_q25', 'nll_q75',
        'bartscore_mean', 'bartscore_q25', 'bartscore_q75',
        'top50_nll_mean', 'top50_bartscore_mean',
        'learning_rate', 'timestamp',
    ]

    def __init__(self, save_dir: str, experiment_name: str, resume: bool = False):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.save_dir / f"metrics_{experiment_name}.csv"
        if not (resume and self.csv_path.exists()):
            with open(self.csv_path, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=self.HEADERS).writeheader()
        print(f"NLP metrics will be saved to: {self.csv_path}")

    def log_epoch(self, metrics: dict):
        from datetime import datetime
        row = {h: metrics.get(h, '') for h in self.HEADERS}
        row['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADERS)
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                             for k, v in row.items()})


# ---------------------------------------------------------------------------
# Scalar stats helper
# ---------------------------------------------------------------------------

def _stats(values: List[float], min_valid: float = 0.0) -> Tuple[float, float, float]:
    """Return (mean, Q25, Q75) from a list, filtering out values below min_valid."""
    valid = [v for v in values if np.isfinite(v) and v >= min_valid]
    if not valid:
        return float('nan'), float('nan'), float('nan')
    arr = np.array(valid, dtype=np.float64)
    return float(arr.mean()), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))


# ---------------------------------------------------------------------------
# BARTScore  (log P(reference | hypothesis) using BART seq2seq)
# ---------------------------------------------------------------------------

_BART_CACHE: dict = {}


def _get_bart_scorer(device: str, model_name: str = 'facebook/bart-base'):
    """Load and cache BartForConditionalGeneration + tokenizer."""
    key = (model_name, device)
    if key not in _BART_CACHE:
        try:
            from transformers import BartForConditionalGeneration, BartTokenizer
            print(f"Loading BARTScore model ({model_name})...")
            tok = BartTokenizer.from_pretrained(model_name)
            mdl = BartForConditionalGeneration.from_pretrained(model_name)
            mdl.eval()
            mdl.to(device)
            _BART_CACHE[key] = (mdl, tok)
            print("BARTScore model loaded.")
        except Exception as e:
            raise RuntimeError(f'Could not load BARTScore model {model_name}') from e
    return _BART_CACHE[key]


@torch.no_grad()
def compute_per_sample_bartscore(
    hypotheses: List[str],
    references: List[str],
    device: str = 'cuda',
    model_name: str = 'facebook/bart-base',
    batch_size: int = 8,
    max_length: int = 128,
) -> List[float]:
    """
    Compute per-sample BARTScore = mean log P(reference_token | hypothesis, ref_prefix).

    Higher (less negative) is better.
    Model-loading and scoring failures raise an error.
    """
    if len(hypotheses) != len(references):
        raise ValueError('Hypotheses and references must have equal lengths')
    scorer = _get_bart_scorer(device, model_name)

    model, tokenizer = scorer
    scores: List[float] = []

    for i in range(0, len(hypotheses), batch_size):
        batch_hyps = hypotheses[i: i + batch_size]
        batch_refs = references[i: i + batch_size]
        try:
            src = tokenizer(
                batch_hyps,
                max_length=max_length, padding='max_length',
                truncation=True, return_tensors='pt',
            )
            tgt = tokenizer(
                batch_refs,
                max_length=max_length, padding='max_length',
                truncation=True, return_tensors='pt',
            )
            input_ids      = src['input_ids'].to(device)
            attention_mask = src['attention_mask'].to(device)
            labels         = tgt['input_ids'].to(device)

            # Replace pad tokens in labels with -100 so they are ignored
            pad_id = tokenizer.pad_token_id
            labels_masked = labels.clone()
            labels_masked[labels_masked == pad_id] = -100

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_masked,
            )
            # output.loss is the mean NLL over the batch; we need per-sample
            logits = output.logits  # [B, T, V]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            for b in range(labels.size(0)):
                lab = labels[b]              # [T]
                valid_mask = lab != pad_id
                if valid_mask.sum() == 0:
                    raise ValueError('BARTScore reference contains no valid tokens')
                gathered = log_probs[b, :, :][
                    torch.arange(lab.size(0), device=device), lab
                ]                            # [T]
                score = gathered[valid_mask].mean().item()
                scores.append(score)
        except Exception as e:
            raise RuntimeError(f'BARTScore failed for batch starting at sample {i}') from e

    return scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_all_metrics(
    hypotheses: List[str],
    references: List[str],
    per_sample_nll: List[float],
    per_sample_bartscore: List[float],
    task: str,
    device: str = 'cpu',
    top_k: int = 50,
) -> Dict[str, float]:
    """
    Aggregate NLP metrics: NLL (mean/Q25/Q75), BARTScore (mean/Q25/Q75),
    top-50 NLL, top-50 BARTScore.

    Args:
        hypotheses: Model-generated texts
        references: Ground-truth texts
        per_sample_nll: Per-sample negative log-likelihoods (lower = better)
        per_sample_bartscore: Per-sample BARTScores (higher = better)
        task: task name (unused, kept for API compat)
        device: torch device
        top_k: number of top samples

    Returns:
        Dictionary of aggregated metric values
    """
    metrics: Dict[str, float] = {}
    if not hypotheses:
        return metrics

    # NLL stats (lower is better → top-50 = lowest NLL)
    nll_mean, nll_q25, nll_q75 = _stats(per_sample_nll)
    metrics['nll_mean'] = nll_mean
    metrics['nll_q25']  = nll_q25
    metrics['nll_q75']  = nll_q75

    # BARTScore stats — values are log-probs (negative), sentinel = -9999.0
    bs_mean, bs_q25, bs_q75 = _stats(per_sample_bartscore, min_valid=-100.0)
    metrics['bartscore_mean'] = bs_mean
    metrics['bartscore_q25']  = bs_q25
    metrics['bartscore_q75']  = bs_q75

    # Top-50 NLL (select samples with lowest NLL)
    n = len(per_sample_nll)
    k = min(top_k, n)
    if k > 0 and any(v >= 0 for v in per_sample_nll):
        nll_sorted = sorted(
            [(v, i) for i, v in enumerate(per_sample_nll) if v >= 0]
        )
        top_nll_idx = [i for _, i in nll_sorted[:k]]
        top_nll_vals = [per_sample_nll[i] for i in top_nll_idx]
        metrics['top50_nll_mean'] = float(np.mean(top_nll_vals))
    else:
        metrics['top50_nll_mean'] = -1.0

    top_indices = sorted((i for i, value in enumerate(per_sample_nll) if value >= 0),
                         key=lambda i: per_sample_nll[i])[:top_k]
    values = [per_sample_bartscore[i] for i in top_indices
              if np.isfinite(per_sample_bartscore[i]) and per_sample_bartscore[i] > -100]
    metrics['top50_bartscore_mean'] = float(np.mean(values)) if values else float('nan')

    return metrics


def compute_per_sample_nlp_metrics(
    hypotheses: List[str],
    references: List[str],
    task: str,
) -> Dict[str, List[float]]:
    """Per-sample BLEU and ROUGE (kept for backward compat / JSON saving)."""
    try:
        from rouge_score import rouge_scorer as rs_mod
        scorer = rs_mod.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    except Exception:
        scorer = None

    per_r1, per_r2, per_rL = [], [], []
    for hyp, ref in zip(hypotheses, references):
        if scorer:
            s = scorer.score(ref, hyp)
            per_r1.append(s['rouge1'].fmeasure)
            per_r2.append(s['rouge2'].fmeasure)
            per_rL.append(s['rougeL'].fmeasure)
        else:
            per_r1.append(-1.0); per_r2.append(-1.0); per_rL.append(-1.0)

    return {
        'per_sample_rouge1': per_r1,
        'per_sample_rouge2': per_r2,
        'per_sample_rougeL': per_rL,
    }


def save_all_samples_nlp_metrics(
    hypotheses: List[str],
    references: List[str],
    per_sample_metrics: Dict[str, List],
    save_dir: str,
    epoch: int,
):
    """Save per-sample metrics and generated texts to a JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'all_samples_epoch_{epoch:04d}.json')
    n = len(hypotheses)
    records = []
    for i in range(n):
        rec = {'idx': i, 'hypothesis': hypotheses[i], 'reference': references[i]}
        for metric_name, values in per_sample_metrics.items():
            if i < len(values):
                rec[metric_name] = values[i]
        records.append(rec)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"All samples metrics saved to: {save_path}")
