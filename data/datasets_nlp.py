"""
NLP Dataset loaders for DAMCHA experiments.

Supports:
- WMT14 (English-German Machine Translation)
- CNN/DailyMail (Summarization)
- CommonGen (Concept-to-Text Generation)
"""

import os
import csv
import tarfile
import hashlib
from typing import Tuple, List, Dict, Optional

import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class Seq2SeqDataset(Dataset):
    """
    Base dataset for sequence-to-sequence tasks.
    Returns (source_ids, source_mask, target_ids, target_mask) tuples.
    """
    
    def __init__(
        self,
        sources: List[str],
        targets: List[str],
        tokenizer,
        max_source_len: int = 512,
        max_target_len: int = 128,
    ):
        """
        Initialize Seq2Seq dataset.
        
        Args:
            sources: List of source texts
            targets: List of target texts
            tokenizer: HuggingFace tokenizer
            max_source_len: Maximum source sequence length
            max_target_len: Maximum target sequence length
        """
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
    
    def __len__(self):
        return len(self.sources)
    
    def __getitem__(self, idx):
        source = self.sources[idx]
        target = self.targets[idx]
        
        # Tokenize source
        source_encoding = self.tokenizer(
            source,
            max_length=self.max_source_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_target_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'source_ids': source_encoding['input_ids'].squeeze(0),
            'source_mask': source_encoding['attention_mask'].squeeze(0),
            'target_ids': target_encoding['input_ids'].squeeze(0),
            'target_mask': target_encoding['attention_mask'].squeeze(0),
            'source_text': source,
            'target_text': target,
        }


def collate_seq2seq(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for Seq2Seq batches.
    """
    source_ids = torch.stack([item['source_ids'] for item in batch])
    source_mask = torch.stack([item['source_mask'] for item in batch])
    target_ids = torch.stack([item['target_ids'] for item in batch])
    target_mask = torch.stack([item['target_mask'] for item in batch])
    source_texts = [item['source_text'] for item in batch]
    target_texts = [item['target_text'] for item in batch]
    
    return {
        'source_ids': source_ids,
        'source_mask': source_mask,
        'target_ids': target_ids,
        'target_mask': target_mask,
        'source_texts': source_texts,
        'target_texts': target_texts,
    }


# =============================================================================
# WMT Chinese-English Dataset
# =============================================================================

def load_wmt_data(data_dir, max_samples=None, split='train'):
    """Read WMT14 English -> German parallel files: WMT14/{split}.en/.de."""
    from pathlib import Path
    folder = Path(data_dir) / 'WMT14'
    sources = (folder / f'{split}.en').read_text(encoding='utf-8').splitlines()
    targets = (folder / f'{split}.de').read_text(encoding='utf-8').splitlines()
    if len(sources) != len(targets) or not sources:
        raise ValueError(f'WMT14 {split}: English and German files must contain equal, nonzero line counts')
    if max_samples is not None:
        sources, targets = sources[:max_samples], targets[:max_samples]
    return sources, targets


def get_wmt_dataloader(data_dir, tokenizer_name='bert-base-multilingual-cased',
                       batch_size=32, max_source_len=128, max_target_len=128,
                       num_workers=4, max_samples=None):
    """Load WMT14 train/validation parallel corpora without re-splitting."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    loaders = []
    for split in ('train', 'validation'):
        sources, targets = load_wmt_data(data_dir, max_samples, split)
        dataset = Seq2SeqDataset(sources, targets, tokenizer, max_source_len, max_target_len)
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=split == 'train',
                                  num_workers=num_workers, collate_fn=collate_seq2seq,
                                  pin_memory=torch.cuda.is_available()))
    return *loaders, tokenizer


# =============================================================================
# CNN/DailyMail Summarization Dataset
# =============================================================================

def _read_text_file(text_file: str) -> List[str]:
    """Read lines from text file."""
    lines = []
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())
    return lines


def _get_url_hashes(path: str) -> Dict[str, bool]:
    """Get hashes of urls in file."""
    urls = _read_text_file(path)
    
    def url_hash(u):
        h = hashlib.sha1()
        try:
            u = u.encode('utf-8')
        except UnicodeDecodeError:
            pass
        h.update(u)
        return h.hexdigest()
    
    return {url_hash(u): True for u in urls}


DM_SINGLE_CLOSE_QUOTE = '\u2019'
DM_DOUBLE_CLOSE_QUOTE = '\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", '`', '"', DM_SINGLE_CLOSE_QUOTE, DM_DOUBLE_CLOSE_QUOTE, ')']


def _get_art_abs(story_file: str) -> Tuple[str, str]:
    """Get article and abstract from story file."""
    lines = _read_text_file(story_file)
    
    def fix_missing_period(line):
        if '@highlight' in line:
            return line
        if not line:
            return line
        if line[-1] in END_TOKENS:
            return line
        return line + ' .'
    
    lines = [fix_missing_period(line) for line in lines]
    
    article_lines = []
    highlights = []
    next_is_highlight = False
    
    for line in lines:
        if not line:
            continue
        elif line.startswith('@highlight'):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)
    
    article = ' '.join(article_lines)
    abstract = '\n'.join(highlights)
    
    return article, abstract


def _get_art_abs_from_content(content: str) -> Tuple[str, str]:
    """Get article and abstract from story file content (string)."""
    lines = content.split('\n')
    lines = [line.strip() for line in lines]
    
    def fix_missing_period(line):
        if '@highlight' in line:
            return line
        if not line:
            return line
        if line[-1] in END_TOKENS:
            return line
        return line + ' .'
    
    lines = [fix_missing_period(line) for line in lines]
    
    article_lines = []
    highlights = []
    next_is_highlight = False
    
    for line in lines:
        if not line:
            continue
        elif line.startswith('@highlight'):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)
    
    article = ' '.join(article_lines)
    abstract = '\n'.join(highlights)
    
    return article, abstract


def load_cnn_dailymail_data(
    data_dir: str,
    split: str = 'train',
    max_samples: int = None,
) -> Tuple[List[str], List[str]]:
    """
    Load CNN/DailyMail summarization data directly from tgz files (no extraction needed).
    
    Args:
        data_dir: Path to data directory
        split: 'train', 'val', or 'test'
        max_samples: Maximum number of samples
        
    Returns:
        Tuple of (articles, summaries)
    """
    cnn_tgz = os.path.join(data_dir, 'CNN_DailyMail', 'cnn_stories.tgz')
    dm_tgz = os.path.join(data_dir, 'CNN_DailyMail', 'dailymail_stories.tgz')
    
    articles = []
    summaries = []
    
    # Load from CNN stories tgz directly
    if os.path.exists(cnn_tgz):
        print(f"Loading CNN stories from tgz (max {max_samples // 2 if max_samples else 'all'} samples)...")
        with tarfile.open(cnn_tgz, 'r:gz') as tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith('.story')]
            for i, member in enumerate(members):
                if max_samples and i >= max_samples // 2:
                    break
                try:
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode('utf-8', errors='ignore')
                        article, summary = _get_art_abs_from_content(content)
                        if article and summary:
                            articles.append(article)
                            summaries.append(summary)
                except Exception:
                    continue
        print(f"  Loaded {len(articles)} CNN stories")
    
    # Load from DailyMail stories tgz directly
    if os.path.exists(dm_tgz):
        cnn_count = len(articles)
        remaining = (max_samples - cnn_count) if max_samples else None
        print(f"Loading DailyMail stories from tgz (max {remaining if remaining else 'all'} samples)...")
        with tarfile.open(dm_tgz, 'r:gz') as tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith('.story')]
            for i, member in enumerate(members):
                if remaining and i >= remaining:
                    break
                try:
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode('utf-8', errors='ignore')
                        article, summary = _get_art_abs_from_content(content)
                        if article and summary:
                            articles.append(article)
                            summaries.append(summary)
                except Exception:
                    continue
        print(f"  Loaded {len(articles) - cnn_count} DailyMail stories")
    
    print(f"Total CNN/DailyMail samples: {len(articles)}")
    return articles, summaries


def get_cnn_dailymail_dataloader(
    data_dir: str,
    tokenizer_name: str = 'facebook/bart-base',
    batch_size: int = 16,
    max_source_len: int = 512,
    max_target_len: int = 128,
    num_workers: int = 4,
    train_split: float = 0.9,
    max_samples: int = 50000,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create CNN/DailyMail data loaders.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load data
    articles, summaries = load_cnn_dailymail_data(data_dir, max_samples=max_samples)
    
    # Split data
    split_idx = int(len(articles) * train_split)
    train_articles, val_articles = articles[:split_idx], articles[split_idx:]
    train_summaries, val_summaries = summaries[:split_idx], summaries[split_idx:]
    
    # Create datasets
    train_dataset = Seq2SeqDataset(
        train_articles, train_summaries, tokenizer,
        max_source_len, max_target_len
    )
    val_dataset = Seq2SeqDataset(
        val_articles, val_summaries, tokenizer,
        max_source_len, max_target_len
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_seq2seq,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_seq2seq,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, tokenizer


# =============================================================================
# CommonGen Dataset
# =============================================================================

def load_commongen_data(
    data_dir: str,
    split: str = 'train',
) -> Tuple[List[str], List[str]]:
    """
    Load CommonGen concept-to-text data.
    
    Args:
        data_dir: Path to data directory
        split: 'train', 'validation', or 'test'
        
    Returns:
        Tuple of (concepts_texts, target_texts)
    """
    parquet_files = {
        'train': 'train-00000-of-00001.parquet',
        'validation': 'validation-00000-of-00001.parquet',
        'test': 'test-00000-of-00001.parquet'
    }
    
    parquet_path = os.path.join(data_dir, 'CommonGen', parquet_files[split])
    
    df = pd.read_parquet(parquet_path)
    
    # Convert concepts list to string
    concepts_texts = []
    target_texts = []
    
    for _, row in df.iterrows():
        concepts = row['concepts']
        if isinstance(concepts, (list, tuple)) or hasattr(concepts, 'tolist'):
            concepts_str = ', '.join(concepts)
        else:
            concepts_str = str(concepts)
        concepts_texts.append(f"Generate a sentence with: {concepts_str}")
        target_texts.append(row['target'])
    
    return concepts_texts, target_texts


def get_commongen_dataloader(
    data_dir: str,
    tokenizer_name: str = 'facebook/bart-base',
    batch_size: int = 32,
    max_source_len: int = 64,
    max_target_len: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create CommonGen data loaders.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load train and validation data
    train_concepts, train_targets = load_commongen_data(data_dir, 'train')
    val_concepts, val_targets = load_commongen_data(data_dir, 'validation')
    
    # Create datasets
    train_dataset = Seq2SeqDataset(
        train_concepts, train_targets, tokenizer,
        max_source_len, max_target_len
    )
    val_dataset = Seq2SeqDataset(
        val_concepts, val_targets, tokenizer,
        max_source_len, max_target_len
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_seq2seq,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_seq2seq,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, tokenizer


# =============================================================================
# Unified Interface
# =============================================================================

def get_nlp_dataloader(
    dataset_name: str,
    data_dir: str,
    tokenizer_name: str = None,
    batch_size: int = 32,
    max_source_len: int = 256,
    max_target_len: int = 128,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, AutoTokenizer]:
    """
    Get NLP dataset by name.
    
    Args:
        dataset_name: 'wmt', 'cnn_dailymail', or 'commongen'
        data_dir: Data directory
        tokenizer_name: HuggingFace tokenizer name or local path (auto-selected if None)
        batch_size: Batch size
        max_source_len: Maximum source length
        max_target_len: Maximum target length
        num_workers: Number of workers
        
    Returns:
        Tuple of (train_loader, val_loader, tokenizer)
    """
    # Check for local BART tokenizer first
    local_bart_path = os.path.join(data_dir, 'BART')
    
    # Default tokenizers for each dataset
    # Use local BART if available, otherwise try HuggingFace
    if os.path.exists(local_bart_path) and os.path.exists(os.path.join(local_bart_path, 'tokenizer.json')):
        default_tokenizer = local_bart_path
    else:
        default_tokenizer = 'facebook/bart-base'
    
    default_tokenizers = {
        'wmt': 'bert-base-multilingual-cased',
        'cnn_dailymail': default_tokenizer,
        'commongen': default_tokenizer,
    }
    
    if tokenizer_name is None:
        tokenizer_name = default_tokenizers.get(dataset_name.lower(), default_tokenizer)
    
    if dataset_name.lower() == 'wmt':
        return get_wmt_dataloader(
            data_dir, tokenizer_name, batch_size,
            max_source_len, max_target_len, num_workers,
            **kwargs
        )
    elif dataset_name.lower() == 'cnn_dailymail':
        return get_cnn_dailymail_dataloader(
            data_dir, tokenizer_name, batch_size,
            max_source_len, max_target_len, num_workers,
            **kwargs
        )
    elif dataset_name.lower() == 'commongen':
        return get_commongen_dataloader(
            data_dir, tokenizer_name, batch_size,
            max_source_len, max_target_len, num_workers
        )
    else:
        raise ValueError(f"Unsupported NLP dataset: {dataset_name}. "
                        f"Supported: ['wmt', 'cnn_dailymail', 'commongen']")


def get_nlp_dataset_info(dataset_name: str) -> dict:
    """
    Get information about an NLP dataset.
    """
    info = {
        'wmt': {
            'task': 'machine_translation',
            'source_lang': 'zh',
            'target_lang': 'en',
            'metrics': ['bleu', 'ter', 'chrf', 'bartscore'],
            'default_max_source_len': 128,
            'default_max_target_len': 128,
        },
        'cnn_dailymail': {
            'task': 'summarization',
            'metrics': ['rouge', 'bartscore'],
            'default_max_source_len': 512,
            'default_max_target_len': 128,
        },
        'commongen': {
            'task': 'concept_to_text',
            'metrics': ['bleu', 'rouge', 'bartscore'],
            'default_max_source_len': 64,
            'default_max_target_len': 64,
        }
    }
    
    return info.get(dataset_name.lower(), {})
