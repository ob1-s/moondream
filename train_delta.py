#!/usr/bin/env python3
"""
Training script for the Delta-Vector VLM extension.

Trains only the DeltaProjector module while keeping the base Moondream model frozen.
Uses the Spot-the-Diff dataset for learning to describe visual changes.

Usage:
    python train_delta.py [--batch_size 8] [--epochs 3] [--lr 3e-4] [--device cuda]
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
import os

# Add moondream to path if running from project root
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from moondream.torch.hf_moondream import HfMoondream
from moondream.torch.delta import ResidualMoondream
from moondream.torch.datasets import SpotTheDiffDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Train Delta-Vector VLM extension")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps (None for full epochs)")
    parser.add_argument("--dry_run", action="store_true", help="Run a few steps for testing")
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.dry_run:
        args.batch_size = 2
        args.max_steps = 10
        args.device = "cpu"
        print("🧪 Dry run mode: batch_size=2, max_steps=10, device=cpu")
    
    print(f"📦 Loading base model on {args.device}...")
    
    # Load base Moondream model via HuggingFace
    # Note: For local development, you might want to use a local path
    try:
        from transformers import AutoModelForCausalLM
        base_hf = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        )
    except Exception as e:
        print(f"⚠️  Could not load from HuggingFace: {e}")
        print("Attempting to load local model...")
        # Fallback for local testing
        from moondream.torch.config import MoondreamConfig
        from moondream.torch.moondream import MoondreamModel
        base_model = MoondreamModel(MoondreamConfig())
        base_hf = type('FakeHF', (), {'model': base_model})()
    
    # Get the underlying MoondreamModel
    if hasattr(base_hf, 'model') and hasattr(base_hf.model, 'config'):
        base_model = base_hf.model
    else:
        base_model = base_hf
    
    # Wrap with ResidualMoondream
    model = ResidualMoondream(base_model, out_tokens=64)
    model = model.to(args.device)
    
    # Verify frozen parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"✅ Trainable parameters: {trainable_params:,}")
    print(f"❄️  Frozen parameters: {frozen_params:,}")
    
    # Get tokenizer from base model
    tokenizer = base_model.tokenizer
    eos_id = base_model.config.tokenizer.eos_id
    
    print("📚 Loading Spot-the-Diff dataset...")
    dataset = SpotTheDiffDataset(tokenizer, split="train")
    print(f"   Dataset size: {len(dataset)} samples")
    
    # Create dataloader
    collate_with_eos = partial(collate_fn, eos_id=eos_id)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_with_eos,
        pin_memory=(args.device == "cuda"),
    )
    
    # Optimizer - only optimize delta_proj parameters
    optimizer = torch.optim.AdamW(
        model.delta_proj.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )
    
    # Learning rate scheduler
    total_steps = len(dataloader) * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_steps,
        eta_min=args.lr * 0.1,
    )
    
    # Create checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\n🚀 Starting training for {args.epochs} epoch(s)...")
    print(f"   Total steps: {total_steps}")
    print(f"   Effective batch size: {args.batch_size * args.gradient_accumulation}")
    
    model.train()
    global_step = 0
    running_loss = 0.0
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            img_prev = batch['img_prev'].to(args.device, dtype=model.dtype)
            img_curr = batch['img_curr'].to(args.device, dtype=model.dtype)
            tokens = batch['tokens'].to(args.device)
            labels = batch['labels'].to(args.device)
            
            # Forward pass
            try:
                loss, logits = model.training_forward(
                    img_prev=img_prev,
                    img_curr=img_curr,
                    text_tokens=tokens,
                    labels=labels,
                )
            except Exception as e:
                print(f"\n⚠️  Error in forward pass: {e}")
                if args.dry_run:
                    raise
                continue
            
            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % args.gradient_accumulation == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.delta_proj.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
            
            # Logging
            running_loss += loss.item() * args.gradient_accumulation
            epoch_loss += loss.item() * args.gradient_accumulation
            num_batches += 1
            
            if global_step % args.log_interval == 0 and global_step > 0:
                avg_loss = running_loss / args.log_interval
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{lr:.2e}",
                    "step": global_step,
                })
                running_loss = 0.0
            
            # Check max steps
            if args.max_steps and global_step >= args.max_steps:
                print(f"\n✅ Reached max_steps ({args.max_steps})")
                break
        
        # End of epoch
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"\n📊 Epoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f"delta_proj_epoch{epoch+1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state_dict": model.delta_proj.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_epoch_loss,
        }, checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")
        
        if args.max_steps and global_step >= args.max_steps:
            break
    
    print("\n✨ Training complete!")
    
    # Final verification
    if args.dry_run:
        print("\n🔍 Verifying gradient flow...")
        has_gradients = any(
            p.grad is not None and p.grad.abs().sum() > 0 
            for p in model.delta_proj.parameters()
        )
        base_has_gradients = any(
            p.grad is not None and p.grad.abs().sum() > 0 
            for p in model.model.parameters()
        )
        
        print(f"   DeltaProjector has gradients: {has_gradients}")
        print(f"   Base model has gradients: {base_has_gradients} (should be False)")
        
        if has_gradients and not base_has_gradients:
            print("   ✅ Gradient flow is correct!")
        else:
            print("   ⚠️  Gradient flow might have issues")


if __name__ == "__main__":
    main()
