"""
Dataset utilities for training the Delta-Vector VLM extension.

Provides data loading for the Spot-the-Diff dataset which contains
pairs of images with descriptions of what changed between them.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

from PIL import Image
from torchvision import transforms


class SpotTheDiffDataset(Dataset):
    """
    Dataset for the Spot-the-Diff task using Lancelot53/spot-the-diff from HuggingFace.
    
    Each sample contains:
    - image_before: The "previous" frame
    - image_after: The "current" frame  
    - sentences: List of descriptions of what changed
    
    Args:
        tokenizer: The Moondream tokenizer for encoding text
        split: Dataset split ("train" or "test")
        max_length: Maximum token length for captions
    """
    
    def __init__(
        self,
        tokenizer,
        split: str = "train",
        max_length: int = 256,
    ):
        if not HAS_DATASETS:
            raise ImportError(
                "The 'datasets' library is required. "
                "Install with: pip install datasets"
            )
        
        self.ds = load_dataset("Lancelot53/spot-the-diff", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Standard Moondream preprocessing: 378x378, normalized to [-1, 1]
        self.preprocess = transforms.Compose([
            transforms.Resize(
                (378, 378), 
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def __len__(self) -> int:
        return len(self.ds)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.ds[idx]
        
        # Load and preprocess images
        img_prev = item['image_before'].convert('RGB')
        img_curr = item['image_after'].convert('RGB')
        
        img_prev_tensor = self.preprocess(img_prev)
        img_curr_tensor = self.preprocess(img_curr)
        
        # Get caption (first sentence describing the change)
        sentences = item.get('sentences', item.get('sentence', []))
        if isinstance(sentences, list) and len(sentences) > 0:
            caption = sentences[0]
        elif isinstance(sentences, str):
            caption = sentences
        else:
            caption = "Something changed."
        
        # Format as Q&A
        # Prompt: "Question: What changed?\n\nAnswer: {caption}"
        prompt = f"\n\nQuestion: What changed?\n\nAnswer: {caption}"
        
        # Tokenize
        tokens = self.tokenizer.encode(prompt).ids
        
        # Truncate if needed
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        return {
            "img_prev": img_prev_tensor,
            "img_curr": img_curr_tensor,
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "caption": caption,
        }


def collate_fn(
    batch: List[Dict[str, Any]], 
    eos_id: int = 0,
    pad_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching SpotTheDiff samples.
    
    Handles:
    - Stacking image tensors
    - Padding token sequences to uniform length
    - Creating label masks (ignoring padding with -100)
    
    Args:
        batch: List of samples from SpotTheDiffDataset
        eos_id: End-of-sequence token ID
        pad_id: Padding token ID
        
    Returns:
        Dictionary with batched tensors
    """
    # Stack images
    img_prevs = torch.stack([b['img_prev'] for b in batch])
    img_currs = torch.stack([b['img_curr'] for b in batch])
    
    # Get max length for padding
    max_len = max(len(b['tokens']) for b in batch)
    
    padded_tokens = []
    labels = []
    attention_masks = []
    
    for b in batch:
        toks = b['tokens']
        seq_len = len(toks)
        pad_len = max_len - seq_len
        
        # Pad tokens
        if pad_len > 0:
            padding = torch.full((pad_len,), pad_id, dtype=torch.long)
            padded = torch.cat([toks, padding])
        else:
            padded = toks
        padded_tokens.append(padded)
        
        # Create labels (ignore padding with -100)
        label_vec = padded.clone()
        if pad_len > 0:
            label_vec[seq_len:] = -100
        labels.append(label_vec)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attn_mask = torch.ones(max_len, dtype=torch.long)
        if pad_len > 0:
            attn_mask[seq_len:] = 0
        attention_masks.append(attn_mask)
    
    return {
        "img_prev": img_prevs,
        "img_curr": img_currs,
        "tokens": torch.stack(padded_tokens),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_masks),
    }


def create_dataloader(
    tokenizer,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    eos_id: int = 0,
    **kwargs,
):
    """
    Create a DataLoader for the SpotTheDiff dataset.
    
    Args:
        tokenizer: Moondream tokenizer
        split: Dataset split
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle the data
        eos_id: End-of-sequence token ID
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        torch.utils.data.DataLoader
    """
    from torch.utils.data import DataLoader
    from functools import partial
    
    dataset = SpotTheDiffDataset(tokenizer, split=split)
    
    collate = partial(collate_fn, eos_id=eos_id)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        **kwargs,
    )
